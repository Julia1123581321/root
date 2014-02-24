//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "ASTNodeEraser.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Utils/AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/DependentDiagnostic.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Sema.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Preprocessor.h"

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JIT.h" // For debugging the EE in gdb
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
//#include "llvm/Transforms/IPO.h"

using namespace clang;

namespace cling {

  ///\brief The class does the actual work of removing a declaration and
  /// resetting the internal structures of the compiler
  ///
  class DeclReverter : public DeclVisitor<DeclReverter, bool> {
  private:
    typedef llvm::DenseSet<FileID> FileIDs;

    ///\brief The Sema object being reverted (contains the AST as well).
    ///
    Sema* m_Sema;

    ///\brief The execution engine, either JIT or MCJIT, being recovered.
    ///
    llvm::ExecutionEngine* m_EEngine;

    ///\brief The current transaction being reverted.
    ///
    const Transaction* m_CurTransaction;

    ///\brief Reverted declaration contains a SourceLocation, representing a
    /// place in the file where it was seen. Clang caches that file and even if
    /// a declaration is removed and the file is edited we hit the cached entry.
    /// This ADT keeps track of the files from which the reverted declarations
    /// came from so that in the end they could be removed from clang's cache.
    ///
    FileIDs m_FilesToUncache;

  public:
    DeclReverter(Sema* S, llvm::ExecutionEngine* EE, const Transaction* T)
      : m_Sema(S), m_EEngine(EE), m_CurTransaction(T) { }
    ~DeclReverter();

    ///\brief Interface with nice name, forwarding to Visit.
    ///
    ///\param[in] D - The declaration to forward.
    ///\returns true on success.
    ///
    bool RevertDecl(Decl* D) { return Visit(D); }

    ///\brief If it falls back in the base class just remove the declaration
    /// only from the declaration context.
    /// @param[in] D - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitDecl(Decl* D);

    ///\brief Removes the declaration from the lookup chains and from the
    /// declaration context.
    /// @param[in] ND - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitNamedDecl(NamedDecl* ND);

    ///\brief Removes a using shadow declaration, created in the cases:
    ///\code
    /// namespace A {
    ///   void foo();
    /// }
    /// namespace B {
    ///   using A::foo; // <- a UsingDecl
    ///                 // Also creates a UsingShadowDecl for A::foo() in B
    /// }
    ///\endcode
    ///\param[in] USD - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitUsingShadowDecl(UsingShadowDecl* USD);

    ///\brief Removes a typedef name decls. A base class for TypedefDecls and
    /// TypeAliasDecls.
    ///\param[in] TND - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitTypedefNameDecl(TypedefNameDecl* TND);

    ///\brief Removes the declaration from the lookup chains and from the
    /// declaration context and it rebuilds the redeclaration chain.
    /// @param[in] VD - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitVarDecl(VarDecl* VD);

    ///\brief Removes the declaration from the lookup chains and from the
    /// declaration context and it rebuilds the redeclaration chain.
    /// @param[in] FD - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitFunctionDecl(FunctionDecl* FD);

    ///\brief Specialize the removal of constructors due to the fact the we need
    /// the constructor type (aka CXXCtorType). The information is located in
    /// the CXXConstructExpr of usually VarDecls. 
    /// See clang::CodeGen::CodeGenFunction::EmitCXXConstructExpr.
    ///
    /// What we will do instead is to brute-force and try to remove from the 
    /// llvm::Module all ctors of this class with all the types.
    ///
    ///\param[in] CXXCtor - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitCXXConstructorDecl(CXXConstructorDecl* CXXCtor);

    ///\brief Removes the DeclCotnext and its decls.
    /// @param[in] DC - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitDeclContext(DeclContext* DC);

    ///\brief Removes the namespace.
    /// @param[in] NSD - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitNamespaceDecl(NamespaceDecl* NSD);

    ///\brief Removes a Tag (class/union/struct/enum). Most of the other
    /// containers fall back into that case.
    /// @param[in] TD - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitTagDecl(TagDecl* TD);

    ///\brief Removes a RecordDecl. We shouldn't remove the implicit class 
    /// declaration.
    ///\param[in] RD - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitRecordDecl(RecordDecl* RD);

    ///\brief Remove the macro from the Preprocessor.
    /// @param[in] MD - The MacroDirectiveInfo containing the IdentifierInfo and
    ///                MacroDirective to forward.
    ///
    ///\returns true on success.
    ///
    bool VisitMacro(const Transaction::MacroDirectiveInfo MD);

    ///@name Templates
    ///@{

    ///\brief Removes template from the redecl chain. Templates are 
    /// redeclarables also.
    /// @param[in] R - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitRedeclarableTemplateDecl(RedeclarableTemplateDecl* R);


    ///\brief Removes the declaration clang's internal structures. This case
    /// looks very much to VisitFunctionDecl, but FunctionTemplateDecl doesn't
    /// derive from FunctionDecl and thus we need to handle it 'by hand'.
    /// @param[in] FTD - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitFunctionTemplateDecl(FunctionTemplateDecl* FTD);

    ///\brief Removes a class template declaration from clang's internal
    /// structures.
    /// @param[in] CTD - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitClassTemplateDecl(ClassTemplateDecl* CTD);

    ///\brief Removes a class template specialization declaration from clang's
    /// internal structures.
    /// @param[in] CTSD - The declaration to be removed.
    ///
    ///\returns true on success.
    ///
    bool VisitClassTemplateSpecializationDecl(ClassTemplateSpecializationDecl* 
                                              CTSD);

    ///@}

    void MaybeRemoveDeclFromModule(GlobalDecl& GD) const;
    void RemoveStaticInit(llvm::Function& F) const;

    /// @name Helpers
    /// @{

    ///\brief Interface with nice name, forwarding to Visit.
    ///
    ///\param[in] MD - The MacroDirectiveInfo containing the IdentifierInfo and
    ///                MacroDirective to forward.
    ///\returns true on success.
    ///
    bool RevertMacro(Transaction::MacroDirectiveInfo MD) { 
      return VisitMacro(MD);
    }

    ///\brief Removes given declaration from the chain of redeclarations.
    /// Rebuilds the chain and sets properly first and last redeclaration.
    /// @param[in] R - The redeclarable, its chain to be rebuilt.
    /// @param[in] DC - Remove the redecl's lookup entry from this DeclContext.
    ///
    ///\returns the most recent redeclaration in the new chain.
    ///
    template <typename T>
    bool VisitRedeclarable(clang::Redeclarable<T>* R, DeclContext* DC) {
      llvm::SmallVector<T*, 4> PrevDecls;
      T* PrevDecl = R->getMostRecentDecl();
      // [0]=>C [1]=>B [2]=>A ...
      while (PrevDecl) { // Collect the redeclarations, except the one we remove
        if (PrevDecl != R)
          PrevDecls.push_back(PrevDecl);
        PrevDecl = PrevDecl->getPreviousDecl();
      }

      if (!PrevDecls.empty()) {
        // Make sure we update the lookup maps, because the removed decl might
        // be registered in the lookup and again findable.
        StoredDeclsMap* Map = DC->getPrimaryContext()->getLookupPtr();
        if (Map) {
          NamedDecl* ND = (NamedDecl*)((T*)R);
          DeclarationName Name = ND->getDeclName();
          if (!Name.isEmpty()) {
            StoredDeclsMap::iterator Pos = Map->find(Name);
            if (Pos != Map->end() && !Pos->second.isNull()) {
              DeclContext::lookup_result decls = Pos->second.getLookupResult();

              for(DeclContext::lookup_result::iterator I = decls.begin(),
                    E = decls.end(); I != E; ++I) {
                // FIXME: A decl meant to be added in the lookup already exists
                // in the lookup table. My assumption is that the DeclReverted
                // adds it here. This needs to be investigated mode. For now
                // std::find gets promoted from assert to condition :)
                if (*I == ND && std::find(decls.begin(), decls.end(), 
                                          PrevDecls[0]) == decls.end()) {
                  // The decl was registered in the lookup, update it.
                  *I = PrevDecls[0];
                  break;
                }
              }
            }
          }
        }
        // Put 0 in the end of the array so that the loop will reset the
        // pointer to latest redeclaration in the chain to itself.
        //
        PrevDecls.push_back(0);

        // 0 <- A <- B <- C
        for(unsigned i = PrevDecls.size() - 1; i > 0; --i) {
          PrevDecls[i-1]->setPreviousDecl(PrevDecls[i]);
        }
      }
      return true;
    }

    /// @}

  private:
    ///\brief Function that collects the files which we must reread from disk.
    ///
    /// For example: We must uncache the cached include, which brought a
    /// declaration or a macro diretive definition in the AST.
    ///\param[in] Loc - The source location of the reverted declaration.
    ///
    void CollectFilesToUncache(SourceLocation Loc);
  };

  DeclReverter::~DeclReverter() {
    SourceManager& SM = m_Sema->getSourceManager();
    for (FileIDs::iterator I = m_FilesToUncache.begin(), 
           E = m_FilesToUncache.end(); I != E; ++I) {
      const SrcMgr::FileInfo& fInfo = SM.getSLocEntry(*I).getFile();
      // We need to reset the cache
      SrcMgr::ContentCache* cache 
        = const_cast<SrcMgr::ContentCache*>(fInfo.getContentCache());
      FileEntry* entry = const_cast<FileEntry*>(cache->ContentsEntry);
      // We have to reset the file entry size to keep the cache and the file
      // entry in sync.
      if (entry) {
        cache->replaceBuffer(0,/*free*/true);
        FileManager::modifyFileEntry(entry, /*size*/0, 0);
      }
    }

    // Clean up the pending instantiations
    m_Sema->PendingInstantiations.clear();
    m_Sema->PendingLocalImplicitInstantiations.clear();
  }

  void DeclReverter::CollectFilesToUncache(SourceLocation Loc) {
    const SourceManager& SM = m_Sema->getSourceManager();
    FileID FID = SM.getFileID(SM.getSpellingLoc(Loc));
    if (!FID.isInvalid() && FID >= m_CurTransaction->getBufferFID()
        && !m_FilesToUncache.count(FID)) 
      m_FilesToUncache.insert(FID);
  }

  bool DeclReverter::VisitDecl(Decl* D) {
    assert(D && "The Decl is null");
    CollectFilesToUncache(D->getLocStart());

    DeclContext* DC = D->getLexicalDeclContext();

    bool Successful = true;
    if (DC->containsDecl(D))
      DC->removeDecl(D);

    // With the bump allocator this is nop.
    if (Successful)
      m_Sema->getASTContext().Deallocate(D);
    return Successful;
  }

  bool DeclReverter::VisitNamedDecl(NamedDecl* ND) {
    bool Successful = VisitDecl(ND);

    DeclContext* DC = ND->getDeclContext();
    while (DC->isTransparentContext())
      DC = DC->getLookupParent();

    // if the decl was anonymous we are done.
    if (!ND->getIdentifier())
      return Successful;

     // If the decl was removed make sure that we fix the lookup
    if (Successful) {
      if (Scope* S = m_Sema->getScopeForContext(DC))
        S->RemoveDecl(ND);

      if (utils::Analyze::isOnScopeChains(ND, *m_Sema))
        m_Sema->IdResolver.RemoveDecl(ND);
    }

    // Cleanup the lookup tables.
    StoredDeclsMap *Map = DC->getPrimaryContext()->getLookupPtr();
    if (Map) { // DeclContexts like EnumDecls don't have lookup maps.
      // Make sure we the decl doesn't exist in the lookup tables.
      StoredDeclsMap::iterator Pos = Map->find(ND->getDeclName());
      if ( Pos != Map->end()) {
        // Most decls only have one entry in their list, special case it.
        if (Pos->second.getAsDecl() == ND)
          Pos->second.remove(ND);
        else if (StoredDeclsList::DeclsTy* Vec = Pos->second.getAsVector()) {
          // Otherwise iterate over the list with entries with the same name.
          for (StoredDeclsList::DeclsTy::const_iterator I = Vec->begin(),
                 E = Vec->end(); I != E; ++I)
            if (*I == ND)
              Pos->second.remove(ND);
        }
        if (Pos->second.isNull() || 
            (Pos->second.getAsVector() && !Pos->second.getAsVector()->size()))
          Map->erase(Pos);
      }
    }

#ifndef NDEBUG
    if (Map) { // DeclContexts like EnumDecls don't have lookup maps.
      // Make sure we the decl doesn't exist in the lookup tables.
      StoredDeclsMap::iterator Pos = Map->find(ND->getDeclName());
      if ( Pos != Map->end()) {
        // Most decls only have one entry in their list, special case it.
        if (NamedDecl *OldD = Pos->second.getAsDecl())
          assert(OldD != ND && "Lookup entry still exists.");
        else if (StoredDeclsList::DeclsTy* Vec = Pos->second.getAsVector()) {
          // Otherwise iterate over the list with entries with the same name.
          // TODO: Walk the redeclaration chain if the entry was a redeclaration.

          for (StoredDeclsList::DeclsTy::const_iterator I = Vec->begin(),
                 E = Vec->end(); I != E; ++I)
            assert(*I != ND && "Lookup entry still exists.");
        }
        else
          assert(Pos->second.isNull() && "!?");
      }
    }
#endif

    return Successful;
  }

  bool DeclReverter::VisitUsingShadowDecl(UsingShadowDecl* USD) {
    // UsingShadowDecl: NamedDecl, Redeclarable
    bool Successful = true;
    // FIXME: This is needed when we have newest clang:
    //Successful = VisitRedeclarable(USD, USD->getDeclContext());
    Successful &= VisitNamedDecl(USD);

    // Unregister from the using decl that it shadows.
    USD->getUsingDecl()->removeShadowDecl(USD);

    return Successful;
  }

  bool DeclReverter::VisitTypedefNameDecl(TypedefNameDecl* TND) {
    // TypedefNameDecl: TypeDecl, Redeclarable
    bool Successful = VisitRedeclarable(TND, TND->getDeclContext());
    Successful &= VisitTypeDecl(TND);
    return Successful;
  }


  bool DeclReverter::VisitVarDecl(VarDecl* VD) {
    // llvm::Module cannot contain:
    // * variables and parameters with dependent context;
    // * mangled names for parameters;
    if (!isa<ParmVarDecl>(VD) && !VD->getDeclContext()->isDependentContext()) {
      // Cleanup the module if the transaction was committed and code was
      // generated. This has to go first, because it may need the AST 
      // information which we will remove soon. (Eg. mangleDeclName iterates the
      // redecls)
      GlobalDecl GD(VD);
      MaybeRemoveDeclFromModule(GD);
    }

    // VarDecl : DeclaratiorDecl, Redeclarable
    bool Successful = VisitRedeclarable(VD, VD->getDeclContext());
    Successful &= VisitDeclaratorDecl(VD);

    return Successful;
  }

  namespace {
    typedef llvm::SmallVector<VarDecl*, 2> Vars;
    class StaticVarCollector : public RecursiveASTVisitor<StaticVarCollector> {
      Vars& m_V;
    public:
      StaticVarCollector(FunctionDecl* FD, Vars& V) : m_V(V) {
        TraverseStmt(FD->getBody());
      }
      bool VisitDeclStmt(DeclStmt* DS) {
        for(DeclStmt::decl_iterator I = DS->decl_begin(), E = DS->decl_end();
            I != E; ++I)
          if (VarDecl* VD = dyn_cast<VarDecl>(*I))
            if (VD->isStaticLocal())
              m_V.push_back(VD);
        return true;
      }
    };
  }
  bool DeclReverter::VisitFunctionDecl(FunctionDecl* FD) {
    // The Structors need to be handled differently.
    if (!isa<CXXConstructorDecl>(FD) && !isa<CXXDestructorDecl>(FD)) {
      // Cleanup the module if the transaction was committed and code was
      // generated. This has to go first, because it may need the AST info
      // which we will remove soon. (Eg. mangleDeclName iterates the redecls)
      GlobalDecl GD(FD);
      MaybeRemoveDeclFromModule(GD);
      // Handle static locals. void func() { static int var; } is represented in
      // the llvm::Module is a global named @func.var
      Vars V;
      StaticVarCollector c(FD, V);
      for (Vars::iterator I = V.begin(), E = V.end(); I != E; ++I) {
        GlobalDecl GD(*I);
        MaybeRemoveDeclFromModule(GD);
      }
    }
    // FunctionDecl : DeclaratiorDecl, DeclContext, Redeclarable
    // We start with the decl context first, because parameters are part of the
    // DeclContext and when trying to remove them we need the full redecl chain
    // still in place.
    bool Successful = VisitDeclContext(FD);
    Successful &= VisitRedeclarable(FD, FD->getDeclContext());
    Successful &= VisitDeclaratorDecl(FD);

    // Template instantiation of templated function first creates a canonical
    // declaration and after the actual template specialization. For example:
    // template<typename T> T TemplatedF(T t);
    // template<> int TemplatedF(int i) { return i + 1; } creates:
    // 1. Canonical decl: int TemplatedF(int i);
    // 2. int TemplatedF(int i){ return i + 1; }
    //
    // The template specialization is attached to the list of specialization of
    // the templated function.
    // When TemplatedF is looked up it finds the templated function and the
    // lookup is extended by the templated function with its specializations.
    // In the end we don't need to remove the canonical decl because, it
    // doesn't end up in the lookup table.
    //
    class FunctionTemplateDeclExt : public FunctionTemplateDecl {
    public:
      static void removeSpecialization(FunctionTemplateDecl* self,
                                       const FunctionDecl* specialization) {
        assert(self && specialization && "Cannot be null!");
        assert(specialization == specialization->getCanonicalDecl()
               && "Not the canonical specialization!?");
        typedef llvm::SmallVector<FunctionDecl*, 4> Specializations;
        typedef llvm::FoldingSetVector< FunctionTemplateSpecializationInfo> Set;

        FunctionTemplateDeclExt* This = (FunctionTemplateDeclExt*) self;
        Specializations specializations;
        const Set& specs = This->getSpecializations();

        if (!specs.size()) // nothing to remove
          return;

        // Collect all the specializations without the one to remove.
        for(Set::const_iterator I = specs.begin(),E = specs.end(); I != E; ++I){
          assert(I->Function && "Must have a specialization.");
          if (I->Function != specialization)
            specializations.push_back(I->Function);
        }

        This->getSpecializations().clear();

        //Readd the collected specializations.
        void* InsertPos = 0;
        FunctionTemplateSpecializationInfo* FTSI = 0;
        for (size_t i = 0, e = specializations.size(); i < e; ++i) {
          FTSI = specializations[i]->getTemplateSpecializationInfo();
          assert(FTSI && "Must not be null.");
          // Avoid assertion on add.
          FTSI->SetNextInBucket(0);
          This->addSpecialization(FTSI, InsertPos);
        }
#ifndef NDEBUG
        const TemplateArgumentList* args
          = specialization->getTemplateSpecializationArgs();
        assert(!self->findSpecialization(args->data(), args->size(),  InsertPos)
               && "Finds the removed decl again!");
#endif
      }
    };

    if (FD->isFunctionTemplateSpecialization() && FD->isCanonicalDecl()) {
      // Only the canonical declarations are registered in the list of the
      // specializations.
      FunctionTemplateDecl* FTD
        = FD->getTemplateSpecializationInfo()->getTemplate();
      // The canonical declaration of every specialization is registered with
      // the FunctionTemplateDecl.
      // Note this might revert too much in the case:
      //   template<typename T> T f(){ return T();}
      //   template<> int f();
      //   template<> int f() { return 0;}
      // when the template specialization was forward declared the canonical
      // becomes the first forward declaration. If the canonical forward
      // declaration was declared outside the set of the decls to revert we have
      // to keep it registered as a template specialization.
      //
      // In order to diagnose mismatches of the specializations, clang 'injects'
      // a implicit forward declaration making it very hard distinguish between
      // the explicit and the implicit forward declaration. So far the only way
      // to distinguish is by source location comparison.
      // FIXME: When the misbehavior of clang is fixed we must avoid relying on
      // source locations
      FunctionTemplateDeclExt::removeSpecialization(FTD, FD);
    }

    return Successful;
  }

  bool DeclReverter::VisitCXXConstructorDecl(CXXConstructorDecl* CXXCtor) {
    // Cleanup the module if the transaction was committed and code was
    // generated. This has to go first, because it may need the AST information
    // which we will remove soon. (Eg. mangleDeclName iterates the redecls)

    // Brute-force all possibly generated ctors.
    // Ctor_Complete            Complete object ctor.
    // Ctor_Base                Base object ctor.
    // Ctor_CompleteAllocating 	Complete object allocating ctor.
    GlobalDecl GD(CXXCtor, Ctor_Complete);
    MaybeRemoveDeclFromModule(GD);
    GD = GlobalDecl(CXXCtor, Ctor_Base);
    MaybeRemoveDeclFromModule(GD);
    GD = GlobalDecl(CXXCtor, Ctor_CompleteAllocating);
    MaybeRemoveDeclFromModule(GD);

    bool Successful = VisitCXXMethodDecl(CXXCtor);
    return Successful;
  }

  bool DeclReverter::VisitDeclContext(DeclContext* DC) {
    bool Successful = true;
    typedef llvm::SmallVector<Decl*, 64> Decls;
    Decls declsToErase;
    // Removing from single-linked list invalidates the iterators.
    for (DeclContext::decl_iterator I = DC->decls_begin();
         I != DC->decls_end(); ++I) {
      declsToErase.push_back(*I);
    }

    for(Decls::iterator I = declsToErase.begin(), E = declsToErase.end();
        I != E; ++I)
      Successful = Visit(*I) && Successful;
    return Successful;
  }

  bool DeclReverter::VisitNamespaceDecl(NamespaceDecl* NSD) {
    // NamespaceDecl: NamedDecl, DeclContext, Redeclarable
    bool Successful = VisitRedeclarable(NSD, NSD->getDeclContext());
    Successful &= VisitDeclContext(NSD);
    Successful &= VisitNamedDecl(NSD);

    return Successful;
  }

  bool DeclReverter::VisitTagDecl(TagDecl* TD) {
    // TagDecl: TypeDecl, DeclContext, Redeclarable
    bool Successful = VisitRedeclarable(TD, TD->getDeclContext());
    Successful &= VisitDeclContext(TD);
    Successful &= VisitTypeDecl(TD);
    return Successful;
  }

  bool DeclReverter::VisitRecordDecl(RecordDecl* RD) {
    if (RD->isInjectedClassName())
      return true;

    /// The injected class name in C++ is the name of the class that
    /// appears inside the class itself. For example:
    ///
    /// \code
    /// struct C {
    ///   // C is implicitly declared here as a synonym for the class name.
    /// };
    ///
    /// C::C c; // same as "C c;"
    /// \endcode
    // It is another question why it is on the redecl chain.
    // The test show it can be either: 
    // ... <- InjectedC <- C <- ..., i.e previous decl or
    // ... <- C <- InjectedC <- ...
    RecordDecl* InjectedRD = RD->getPreviousDecl();
    if (!(InjectedRD && InjectedRD->isInjectedClassName())) {
      InjectedRD = RD->getMostRecentDecl();
      while (InjectedRD) {
        if (InjectedRD->isInjectedClassName() 
            && InjectedRD->getPreviousDecl() == RD)
          break;
        InjectedRD = InjectedRD->getPreviousDecl(); 
      }
    }

    bool Successful = true;
    if (InjectedRD) {
      assert(InjectedRD->isInjectedClassName() && "Not injected classname?");
      Successful &= VisitRedeclarable(InjectedRD, InjectedRD->getDeclContext());
    }

    Successful &= VisitTagDecl(RD);
    return Successful;
  }

  void DeclReverter::MaybeRemoveDeclFromModule(GlobalDecl& GD) const {
    if (!m_CurTransaction->getModule()) // syntax-only mode exit
      return;
    using namespace llvm;
    // if it was successfully removed from the AST we have to check whether
    // code was generated and remove it.

    // From llvm's mailing list, explanation of the RAUW'd assert:
    //
    // The problem isn't with your call to
    // replaceAllUsesWith per se, the problem is that somebody (I would guess
    // the JIT?) is holding it in a ValueMap.
    //
    // We used to have a problem that some parts of the code would keep a
    // mapping like so:
    //    std::map<Value *, ...>
    // while somebody else would modify the Value* without them noticing,
    // leading to a dangling pointer in the map. To fix that, we invented the
    // ValueMap which puts a Use that doesn't show up in the use_iterator on
    // the Value it holds. When the Value is erased or RAUW'd, the ValueMap is
    // notified and in this case decides that's not okay and terminates the
    // program.
    //
    // Probably what's happened here is that the calling function has had its
    // code generated by the JIT, but not the callee. Thus the JIT emitted a
    // call to a generated stub, and will do the codegen of the callee once
    // that stub is reached. Of course, once the JIT is in this state, it holds
    // on to the Function with a ValueMap in order to prevent things from
    // getting out of sync.
    //
    if (m_CurTransaction->getState() == Transaction::kCommitted) {
      std::string mangledName;
      utils::Analyze::maybeMangleDeclName(GD, mangledName);

      // Handle static locals. void func() { static int var; } is represented in
      // the llvm::Module is a global named @func.var
      if (const VarDecl* VD = dyn_cast<VarDecl>(GD.getDecl()))
        if (VD->isStaticLocal()) {
          std::string functionMangledName;
          GlobalDecl FDGD(cast<FunctionDecl>(VD->getDeclContext()));
          utils::Analyze::maybeMangleDeclName(FDGD, functionMangledName);
          mangledName = functionMangledName + "." + mangledName;
        }

      GlobalValue* GV
        = m_CurTransaction->getModule()->getNamedValue(mangledName);
      if (GV) { // May be deferred decl and thus 0
        // createGVExtractionPass - If deleteFn is true, this pass deletes
        // the specified global values. Otherwise, it deletes as much of the
        // module as possible, except for the global values specified.
        //
        //std::vector<GlobalValue*> GVs;
        //GVs.push_back(GV);
        //llvm::ModulePass* GVExtract = llvm::createGVExtractionPass(GVs, true);
        //GVExtract->runOnModule(*m_CurTransaction->getModule());

        GV->removeDeadConstantUsers();
        if (!GV->use_empty()) {
          // Assert that if there was a use it is not coming from the explicit
          // AST node, but from the implicitly generated functions, which ensure
          // the initialization order semantics. Such functions are:
          // _GLOBAL__I* and __cxx_global_var_init*
          //
          // We can 'afford' to drop all the references because we know that the
          // static init functions must be called only once, and that was
          // already done.
          SmallVector<User*, 4> uses;

          for(llvm::Value::use_iterator I = GV->use_begin(), E = GV->use_end();
              I != E; ++I) {
            uses.push_back(*I);
          }

          for(SmallVector<User*, 4>::iterator I = uses.begin(), E = uses.end();
              I != E; ++I)
            if (llvm::Instruction* instr = dyn_cast<llvm::Instruction>(*I)) {
              llvm::Function* F = instr->getParent()->getParent();
              if (F->getName().startswith("__cxx_global_var_init"))
                RemoveStaticInit(*F);
          }
        }

        // Cleanup the jit mapping of GV->addr.
        m_EEngine->updateGlobalMapping(GV, 0);
        GV->dropAllReferences();
        if (!GV->use_empty()) {
          if (Function* F = dyn_cast<Function>(GV)) {
            Function* dummy
              = Function::Create(F->getFunctionType(), F->getLinkage());
            F->replaceAllUsesWith(dummy);
          }
          else
            GV->replaceAllUsesWith(UndefValue::get(GV->getType()));
        }
        GV->eraseFromParent();
      }
    }
  }

  void DeclReverter::RemoveStaticInit(llvm::Function& F) const {
    // In our very controlled case the parent of the BasicBlock is the
    // static init llvm::Function.
    assert(F.getName().startswith("__cxx_global_var_init")
           && "Not a static init");
    assert(F.hasInternalLinkage() && "Not a static init");
    // The static init functions have the layout:
    // declare internal void @__cxx_global_var_init1() section "..."
    //
    // define internal void @_GLOBAL__I_a2() section "..." {
    // entry:
    //  call void @__cxx_global_var_init1()
    //  ret void
    // }
    //
    assert(F.hasOneUse() && "Must have only one use");
    // erase _GLOBAL__I* first
    llvm::BasicBlock* BB = cast<llvm::Instruction>(F.use_back())->getParent();
    BB->getParent()->eraseFromParent();
    F.eraseFromParent();
  }

  bool DeclReverter::VisitMacro(Transaction::MacroDirectiveInfo MacroD) {
    assert(MacroD.m_MD && "The MacroDirective is null");
    assert(MacroD.m_II && "The IdentifierInfo is null");
    CollectFilesToUncache(MacroD.m_MD->getLocation());

    Preprocessor& PP = m_Sema->getPreprocessor();
#ifndef NDEBUG
    bool ExistsInPP = false;
    // Make sure the macro is in the Preprocessor. Not sure if not redundant
    // because removeMacro looks for the macro anyway in the DenseMap Macros[]
    for (Preprocessor::macro_iterator
           I = PP.macro_begin(/*IncludeExternalMacros*/false),
           E = PP.macro_end(/*IncludeExternalMacros*/false); E !=I; ++I) {
      if ((*I).first == MacroD.m_II) {
        // FIXME:check whether we have the concrete directive on the macro chain
        // && (*I).second == MacroD.m_MD
        ExistsInPP = true;
        break;
      }
    }
    assert(ExistsInPP && "Not in the Preprocessor?!");
#endif

    const MacroDirective* MD = MacroD.m_MD;
    // Undef the definition
    const MacroInfo* MI = MD->getMacroInfo();

    // If the macro is not defined, this is a noop undef, just return.
    if (MI == 0)
      return false;

    // Remove the pair from the macros
    PP.removeMacro(MacroD.m_II, const_cast<MacroDirective*>(MacroD.m_MD));

    return true;
  }

  bool DeclReverter::VisitRedeclarableTemplateDecl(RedeclarableTemplateDecl* R){
    // RedeclarableTemplateDecl: TemplateDecl, Redeclarable
    bool Successful = VisitRedeclarable(R, R->getDeclContext());
    Successful &= VisitTemplateDecl(R);
    return Successful;
  }

  bool DeclReverter::VisitFunctionTemplateDecl(FunctionTemplateDecl* FTD) {
    bool Successful = true;

    // Remove specializations:
    for (FunctionTemplateDecl::spec_iterator I = FTD->spec_begin(), 
           E = FTD->spec_end(); I != E; ++I)
      Successful &= Visit(*I);

    Successful &= VisitRedeclarableTemplateDecl(FTD);
    Successful &= VisitFunctionDecl(FTD->getTemplatedDecl());
    return Successful;
  }

  bool DeclReverter::VisitClassTemplateDecl(ClassTemplateDecl* CTD) {
    // ClassTemplateDecl: TemplateDecl, Redeclarable
    bool Successful = true;
    // Remove specializations:
    for (ClassTemplateDecl::spec_iterator I = CTD->spec_begin(), 
           E = CTD->spec_end(); I != E; ++I)
      Successful &= Visit(*I);

    Successful &= VisitRedeclarableTemplateDecl(CTD);
    Successful &= Visit(CTD->getTemplatedDecl());
    return Successful;
  }

  bool DeclReverter::VisitClassTemplateSpecializationDecl(
                                        ClassTemplateSpecializationDecl* CTSD) {

    // A template specialization is attached to the list of specialization of
    // the templated class.
    //
    class ClassTemplateDeclExt : public ClassTemplateDecl {
    public:
      static void removeSpecialization(ClassTemplateDecl* self,
                                       ClassTemplateSpecializationDecl* spec) {
        assert(self && spec && "Cannot be null!");
        assert(spec == spec->getCanonicalDecl()
               && "Not the canonical specialization!?");
        typedef llvm::SmallVector<ClassTemplateSpecializationDecl*, 4> Specializations;
        typedef llvm::FoldingSetVector<ClassTemplateSpecializationDecl> Set;

        ClassTemplateDeclExt* This = (ClassTemplateDeclExt*) self;
        Specializations specializations;
        Set& specs = This->getSpecializations();

        if (!specs.size()) // nothing to remove
          return;

        // Collect all the specializations without the one to remove.
        for(Set::iterator I = specs.begin(),E = specs.end(); I != E; ++I){
          if (&*I != spec)
            specializations.push_back(&*I);
        }

        This->getSpecializations().clear();

        //Readd the collected specializations.
        void* InsertPos = 0;
        ClassTemplateSpecializationDecl* CTSD = 0;
        for (size_t i = 0, e = specializations.size(); i < e; ++i) {
          CTSD = specializations[i];
          assert(CTSD && "Must not be null.");
          // Avoid assertion on add.
          CTSD->SetNextInBucket(0);
          This->AddSpecialization(CTSD, InsertPos);
        }
      }
    };

    ClassTemplateSpecializationDecl* CanonCTSD =
      static_cast<ClassTemplateSpecializationDecl*>(CTSD->getCanonicalDecl());
    ClassTemplateDeclExt::removeSpecialization(CTSD->getSpecializedTemplate(),
                                               CanonCTSD);
    // ClassTemplateSpecializationDecl: CXXRecordDecl, FoldingSet
    return VisitCXXRecordDecl(CTSD);
  }


  ASTNodeEraser::ASTNodeEraser(Sema* S, llvm::ExecutionEngine* EE)
    : m_Sema(S), m_EEngine(EE) {
  }

  ASTNodeEraser::~ASTNodeEraser() {
  }

  bool ASTNodeEraser::RevertTransaction(Transaction* T) {
    DeclReverter DeclRev(m_Sema, m_EEngine, T);
    bool Successful = true;

    for (Transaction::const_reverse_iterator I = T->rdecls_begin(),
           E = T->rdecls_end(); I != E; ++I) {
      if ((*I).m_Call != Transaction::kCCIHandleTopLevelDecl)
        continue;
      const DeclGroupRef& DGR = (*I).m_DGR;

      for (DeclGroupRef::const_iterator
             Di = DGR.end() - 1, E = DGR.begin() - 1; Di != E; --Di) {
        // Get rid of the declaration. If the declaration has name we should
        // heal the lookup tables as well
        Successful = DeclRev.RevertDecl(*Di) && Successful;
#ifndef NDEBUG
        assert(Successful && "Cannot handle that yet!");
#endif
      }
    }

    for (Transaction::const_reverse_macros_iterator MI = T->rmacros_begin(),
           ME = T->rmacros_end(); MI != ME; ++MI) {
      // Get rid of the macro definition
      Successful = DeclRev.RevertMacro(*MI) && Successful;
#ifndef NDEBUG
      assert(Successful && "Cannot handle that yet!");
#endif
    }


    m_Sema->getDiagnostics().Reset();
    m_Sema->getDiagnostics().getClient()->clear();

    // Cleanup the module from unused global values.
    // if (T->getModule()) {
    //   llvm::ModulePass* globalDCE = llvm::createGlobalDCEPass();
    //   globalDCE->runOnModule(*T->getModule());
    // }
    if (Successful)
      T->setState(Transaction::kRolledBack);
    else
      T->setState(Transaction::kRolledBackWithErrors);

    return Successful;
  }
} // end namespace cling
