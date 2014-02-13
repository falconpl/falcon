/*
   FALCON - The Falcon Programming Language.
   FILE: parsercontext.h

   Compilation context for Falcon source file compilation.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 16 Apr 2011 18:17:13 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_PARSERCONTEXT_H
#define	_FALCON_PARSERCONTEXT_H

#include <falcon/setup.h>
#include <falcon/synfunc.h>
#include <falcon/statement.h>

namespace Falcon {

class String;
class Symbol;

class SourceParser;
class SynFunc;
class Expression;
class ExprSymbol;
class Statement;
class FalconClass;
class Requirement;
class ImportDef;
class ExprLit;
class SymbolMap;

/** Compilation context for Falcon source file compiler (ABC).

 The compilation context is meant to keep track of the progress of the compilation
 process, and also provide extended information about what is being compiled.

 For example, it keeps track of the syntax tree being formed, and of the current
 depth in nested structures being parsed.

 Also, it is notified about new items being created by the grammar rules; for
 example, it is responsible to create new symbols that are then inserted in the
 syntax tree being constructed.

 Notice that this is a base abastract class, as the actual behavior excited by
 the invocation of new items depends on the compilation targets: the Virtual Machine,
 the interactive mode or a Module. The subclasses may create the structures for a
 later review (i.e. extern symbols in modules), or interrupt the parser to
 prompt the user for new data (the interactive compiler).

 \note Before any operation, call openMain
 */
class FALCON_DYN_CLASS ParserContext {
public:

   /** Creates the compiler context.
    \param sp Pointer to the source parser that is using this context.
    \note Before any operation, call openMain
   */
   ParserContext( SourceParser *sp );
   virtual ~ParserContext();

   /** Called when the input terminates.
    This method is called by the compiler when its input is terminated.

    This might be used by this context to check for context scoping errors
    at the end of file.
    */
   virtual void onInputOver() = 0;

   /** Called back when creating a new function.
      \param function The function that is being created.

    This method is called back when a new fuction is being
    created.

    If the function has also a global name, then onGlobalDefined will be called
    \b after onNewFunc; subclasses may cache the function name to understand
    that the new global is referring to this function object.

    \note In case of anonymous function created, this method will be called
    with a entity having an empty name. The subclass is responsible to find
    an adequate name for the entity and set it, if necessary.
    */
   virtual bool onOpenFunc( Function* function ) = 0;

   virtual void onOpenMethod( Class* cls, Function* function ) = 0;

   virtual void onCloseFunc( Function* function ) = 0;

   /** Called back when creating a new class.
      \param cls The variable that is being created.
      \param bIsObj True if the class has been defined as a singleton object
    \param gs A global symbol associated with non-anonymous classes.

    This method is called back when a new class is being
    created.

    If the class has also a global name, then onGlobalDefined will be called
    \b after onNewClass; subclasses may cache the function name to understand
    that the new global is referring to this function object.

    \note In case of anonymous function created, this method will be called
    with a entity having an empty name. The subclass is responsible to find
    an adequate name for the entity and set it, if necessary.
    */
   virtual bool onOpenClass( Class* cls, bool bIsObj ) = 0;

   virtual void onCloseClass( Class* cls, bool bIsObj ) = 0;

   /** Called back when creating a new class.
      \param stmt The statement being created.

    This method is called back when a new statement is being created.
    If the context is empty, then this is a top-level statement.

    This method is called after any other onNew* that may be in effect,
    and also after that the effects on the current context have been
    taken into account (i.e., after an "end", the block statement is closed
    and the context is updated, then onNewStatement is called).

    This method is called also for inner statements (i.e. statements inside other statements),
    so parent statements gets notified throught this callback after their children
    statements.
    */
   virtual void onNewStatement( TreeStep* stmt ) = 0;

   /** Called back when parsing a "load" directive.
      \param path The load parameter
      \param isFsPath True if the path is a filesystem path, false if it's a logical module name.

    */
   virtual void onLoad( const String& path, bool isFsPath ) = 0;

   /** Called back when parsing an "import symbol from name in/as name" directive.
      \param path The module from where to import the symbol.
      \param isFsPath True if the path is a filesystem path, false if it's a
             logical module name.
      \param symName the name of the symbol to import (might be empty for "all").
      \param targetName Name under which the symbol must be aliased or namespace.
      \param bIsNS If true, targetName is to be intended as a namespace.

    This directive imports one symbol (or asks for all the symbols to be
    imported) from a given module. If a symName is given, it is possible to
    specify an asName, indicating an alias for a specific name. If an inName is
    given, then the symbol(s) will be put in the required namespace, otherwise
    they will be put in in a namespace determined using the import path.

    When using the grammar
    @code
      import sym1, sym2, ... symN from ...
    @endcode

    this will generate multiple calls to this method, one per each delcared symbol.

    \note if symName is empt, then asName cannot be specificed. The callee may
    wish to abort with error if it's done.
    */
   virtual bool onImportFrom( ImportDef* def ) = 0;

   /** Called back when parsing an "export symbol" directive.
      \param symName the name of the symbol to export, or an empty string
       for all.

      Specify a symbol to be exported to the global namespace.

      When using the grammar
      @code
         export sym1, sym2, ... symN
      @endcode
    multiple calls to this method will be generated, one per declared symbol.

    When the symName parameter is an empty string, then all the symbols in this
    module should be exported.

    If the symbol name is not found in the module global symbol table, after
    the parsing is complete, then an error should be raised.

    Export directive is meaningless in interactive compilation.
    */
   virtual void onExport(const String& symName) = 0;

   /** Called back when parsing a "directive name = value" directive.
      \param name The name of the directive.
      \param value The value of the directive.
    */
   virtual void onDirective(const String& name, const String& value) = 0;

   /** Invoked when an attribute is found.
    *
    * If the attribute declaration is happening in a matnra,
    * the mantra parameter is non zero.
    *
    * The generator can never be zero.
    *
    * \return false if the attribute is duplicated. Add directly an error
    * if there is any other problem with the attribute, and then return true.
    */
   virtual bool onAttribute(const String& name, TreeStep* generator, Mantra* mantra ) = 0;

   /** Called back when parsing a "global name" directive.
      \param name The symbol that is being imported in the local context.
    */
   virtual void onGlobal( const String& name ) = 0;

   /** Notifies the creation request for a global symbol.
    \param The name of the symbol that is defined.
    \param alreadyDef Set to true if the symbol was already defined.

    This method is called back when the parser sees a symbol being defined,
    but doesn't have a local symbol table where to create it as a local symbol.

    The parser owner has then the ability to create a global symbol that shall
    be inserted in its own symbol table, or return an already existing symbol
    from its table.
    */
   virtual void onGlobalDefined( const String& name, bool &alreadyDef ) = 0;

   /** Notifies the access request for a global symbol.
       \param The name of the symbol that is defined.
       \return true if the variable is local, false if it's extern.

       This method is called back when the parser sees a symbol being accessed,
       but doesn't have a local variable table where to create it as a local variable.

       The subclass should either return an already known global variable,
       create an external variable (or return an already know one), and hence a
       request for external linkage, or eventually generate
       an undefined variable error to the source parser if it can do so.

       \note in case an error is added, the method can return 0.
       */
   virtual bool onGlobalAccessed( const String& name ) = 0;
   
   virtual Item* getValue( const String& name );
   virtual Item* getValue( const Symbol* name ) = 0;

   /** Called back when an international string is found in the code.
    *
    * This offers the implementation the occasion to save the i-string in a table.
    */
   virtual void onIString( const String& string ) = 0;


   /** Opens the main context frame.
    This context frame (main or base context frame) refers to the topmost
    context frame in the source, which doesn't refer to a specific function.

    The compiler or the compiler owner should call this method before starting
    any parsing to set the place where new statements will be added.

    \note Alternately, you could push the main() function context
    frame where appropriate by calling openFunc.
    */
   void openMain(SynTree* main);

   /** Creates a new variable and defines it immediately in the current context.
    \param variable A Variable or symbol name to be created.
    \return A new or already existing symbol.

    This might create a new variable or access an already existing variable
    in the current context.
    */
   void defineSymbol( const String& variable );
   
   /**
    * \return true if the symbol is local, false if it's extern.
    */
   bool accessSymbol( const String& variable );

   /** Defines the symbols that are declared in expressions as locally defined.
    \param expr The branch of the expressions where symbols are defined.
    \see checkSymbols()
    \see defineSymbol()
    */
   void defineSymbols( Expression* expr );

   void accessSymbols( Expression* expr );

   /** Adds a statement to the current context.
    \param stmt The statement to be added.
    \see checkSymbols();
    */
   void addStatement( TreeStep* stmt );

   /** Opens a new block-statement context.
    \param Statement parent The parent that is opening this context.
    \param branch The SynTree under which to add new
    \param bAutoClose Close automatically the context at first statement.
    \see checkSymbols();
    */
   void openBlock( TreeStep* parent, SynTree* branch, bool bAutoClose = false, bool bAutoAdd = true );

   /** Changes the branch of a block statement context without closing it.
    \return A new SynTree if it's possible to open a branch now, 0 otherwise.

    This is useful when switching branch in switches or if/else multiple
    block statements without disrupting the block structure.

    In case of errors in the current branch opening (i.e. because of undefined
    symbols) the method will return null, otherwise it will return a new
    SynTree that can be used to be stored in the opened branch of the
    current statement.

    \note The parent statement of this block stays the originally pushed statement,
      but bstmt is used to check for undefined symbols, as the
    \see checkSymbols();
    */
   SynTree* changeBranch();
   void changeBranch( SynTree* );

   /** Opens a new Function statement context.
    \param func The function being created.

    Anonymous function have a name, but they are not associated with a symbol;
    global functions are associated with a global symbol, that must have been
    already declared somewhere.

    The symbol, if provided, is given back to onNewFunction callback
    in the compiler context when the function is closed.
    */
   void openFunc( SynFunc *func, bool bIsStatic = false );

   /** Opens a new Class statement context.
    \param cls The class being created.
    \param gs An optional global symbol associated with the function.
    \param bIsObject True if this class is declared as singleton object.

    Anonymous classe have a name, but they are not associated with a symbol;
    global classes are associated with a global symbol, that must have been
    already declared somewhere.

    The symbol, if provided, is given back to onNewClass callback
    in the compiler context when the function is closed.
    */
   void openClass( Class *cls, bool bIsObject );

   /** Pops the current context.
    When a context is completed, it's onNew*() method is called by this method.
    */
   void closeContext();

   /** Pops the current context after an error.
    closes a context without calling the onNew* methods.
    */
   void dropContext();

   /** Gets the current syntactic tree.
      \return the current syntactic tree where statements are added, or 0 for none.
    */
   SynTree* currentTree() const { return m_st; }

   /** Gets the current statement.
      \return The statement that is parent to this one, or 0 for nothing.

      Zero is returned also for topmost-level statements inside functions.
      The returned value is non-zero only if a block statement has been opened.
    */
   TreeStep* currentStmt() const { return m_cstatement; }

   /** Gets the current function.
      \return The current function or 0 if the parser is parsing in the main code.
    */
   SynFunc* currentFunc() const { return m_cfunc; }

   /** Gets the current class.
      \return The current class, or 0 if the parser is not inside a class.
    */
   FalconClass* currentClass() const { return m_cclass; }

   /** To be called back by rules when the parser state needs to be pushed.
    Rules must call back this method after having pushed the new state in the
    parsers and the new context in this class.

    In this way, as the context is popped, the parser state is automatically
    popped as well.
    */
   void onStatePushed( bool isPushedState );

   virtual void onStatePopped();

   /** Finds a symbol in one of the existing symbol table.
      \param name The name of a symbol to be searched.
      \return true if the symbol is local.
    The returned symbol is either a local symbol of the topmost symbol table,
    a closed symbol of intermediate symbol tables or a global symbol in the
    lowest symbol table.

    More precisely, if the symbol is found in the topmost table, it is returned
    as-is, while if it's found in an underlying table, it is returned as-is unless
    it's a LocalSymbol. In that case, a new ClosedSymbol is created and added
    to the topmost table before being returned.

    */
   bool isLocalSymbol( const String& name );

   bool isParentLocal( const String& name );

   /** Return true if the current statements are at syntactic top-level.
   \return true if the current statements are "main code" of the current syntax
         context.

    This is true if no class, function or other syntactic bounding construct
    have been opened.

    */
   bool isTopLevel() const;

   bool isCompleteStatement() const { return isTopLevel() && m_cstatement == 0; }

   /** Clear the current parser context. */
   virtual void reset();
   
   void openTempBlock( SynTree* oldBranch, SynTree* newBranch );

   /**
    Opens a literal context.
    
    When the count of opened literal context is > 0, the symbols are
    not searched in in the context symbol tables, but are always generated
    as dynamic and stored in the topmost literal context for garbage collection.
    
    */
   void openLitContext( ExprLit* lit );
   /**
    Pops a previously opened literal context.
    */
   ExprLit* closeLitContext();
   
   /** Returns the current opened literal context, if any, or 0.
    */
   ExprLit* currentLitContext();
   
   /** 
    True if currently we're in a literal context.
    */
   bool isLitContext() const;
   
   bool isGlobalContext() const;

private:
   class CCFrame; // forward decl for Context Frames.

   SourceParser* m_parser;

   // Pre-cached syntree for performance.
   SynTree* m_st;

   // Current function, precached for performance.
   TreeStep* m_cstatement;

   // Current function, precached for performance.
   SynFunc * m_cfunc;

   // Current class, precached for performance.
   FalconClass * m_cclass;

   void saveStatus( CCFrame& cf ) const;
   void restoreStatus( const CCFrame& cf );

   class Private;
   ParserContext::Private* _p;
};

}

#endif	/* _FALCON_PARSERCONTEXT_H */

/* end of parsercontext.h */
