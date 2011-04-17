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

#include "synfunc.h"
#include "statement.h"

namespace Falcon {

class String;
class Symbol;
class GlobalSymbol;
class UnknownSymbol;

class SourceParser;
class SynFunc;
class Expression;
class Statement;
class Class;

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
class ParserContext {
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
    */
   virtual void onNewFunc( Function* function ) = 0;

   /** Called back when creating a new class.
      \param cls The variable that is being created.
      \param bIsObj True if the class has been defined as a singleton object

    This method is called back when a new class is being
    created.

    If the class has also a global name, then onGlobalDefined will be called
    \b after onNewClass; subclasses may cache the function name to understand
    that the new global is referring to this function object.
    */
   virtual void onNewClass( Class* cls, bool bIsObj ) = 0;

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
   virtual void onNewStatement( Statement* stmt ) = 0;

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
      \param asName Name under which the symbol must be aliased Will be nothing
               if not given.
      \param inName Name for the local namespace where the symbol is imported.

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
   virtual void onImportFrom( const String& path, bool isFsPath, const String& symName,
         const String& asName, const String &inName ) = 0;

   /** Called back when parsing an "import symbol " directive.
      \param symName the name of the symbol to import.
    
      Specify a symbol to import from the global namespace.

      When using the grammar
      @code
         import sym1, sym2, ... symN
      @endcode
    multiple calls to this method will be generated, one per declared symbol.
    */
   virtual void onImport(const String& symName ) = 0;

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

   /** Called back when parsing a "global name" directive.
      \param name The symbol that is being imported in the local context.
    */
   virtual void onGlobal( const String& name ) = 0;

   /** Notifies the creation of an external or undefined symbol.
    \param The name of the symbol that is currently undefined.
    \return A new symbol that can be used to form the sequence.
    
    This method is called back when the parser finds an unreferenced symbol
    name. The subclass has here the chance to create a symbol that will need
    to be implicitly imported, and return the instance of the symbol created
    this way, or return a pre-defined symbol that it knows about.
    
    It can also return 0; in that case, the parser will raise an undefined
    symbol error at current location, and onUnkownSymbol is called.

    */
   virtual Symbol* onUndefinedSymbol( const String& name ) = 0;

   /** Notifies the creation request for a global symbol.
    \param The name of the symbol that is defined.
    \return A new symbol that can be used to form the sequence.

    This method is called back when the parser sees a symbol being defined,
    but doesn't have a local symbol table where to create it as a local symbol.

    The parser owner has then the ability to create a global symbol that shall
    be inserted in its own symbol table, or return an already existing symbol
    from its table.

    If the owner returns zero, onUnknownSymbol is called.
    */
   virtual Symbol* onGlobalDefined( const String& name ) = 0;

   /** Called back when any try to define a symbol fail.
    \param sym A symbol to be disposed of.

    Unknown symbols are symbols that cannot be placed in any symbol table,
    because the resolution tries (joint effort of the SourceParser, this class,
    its subclasses and eventually some other entity hold by the subclasses) have
    falied.

    The sym parameter should be stored somewhere to be later disposed, when the
    syntree holding the symbol is not needed anymore. Normally, this is done
    through symbol tables, but this can't be done with unknown symbols as they
    cannot be placed in them.

    The call of this method is always preceded by the queueing of an error
    condition in the parser.
    If the callee is sure that the structure being created won't be ispected
    at a later step, (i.e. because the error result in the parser prevents it),
    then it can destory the symbol immediately, otherwise it should store it
    together with the syntree root, and destroy all them after the syntree
    has gone out of scope.

    */
   virtual void onUnknownSymbol( UnknownSymbol* sym ) = 0;

   /** Opens the main context frame.
    This context frame (main or base context frame) refers to the topmost
    context frame in the source, which doesn't refer to a specfic function.

    The compiler or the compiler owner should call this method before starting
    any parsing to set the place where new statements will be added.
    
    \note Alternately, you could push the main() function context
    frame where appropriate by calling openFunc.
    */
   void openMain(SynTree* main);

   /** Creates a new variable in the current context.
    \param variable A Variable or symbol name to be created.
    \return A new or already existing symbol.

    This might create a new variable or access an already existing variable
    in the current context.

    \note By default, variables are added also as "extern".
    */
   Symbol* addVariable( const String& variable );

   /** Defines the symbols that are declared in expressions as locally defined.
    \param expr The branch of the expressions where symbols are defined.
    \see checkSymbols()
    \see defineSymbol()
    */
   void defineSymbols( Expression* expr );

   /** Define a single symbol (if unknown).
    \param uks A symbol that, if unknown, shall be defined as created in the local context.
    */
   void defineSymbol( Symbol* uks );

   /** Checks the symbols that have been declared up to date.

    Falcon defines symbols by assignment, or trhough particular expressions
    which explicitly declare some symbols. Unknown symbols are implicitly declared
    as extern.

    As symbols are created by the parser, they are temporarily stored in this
    parser context. Some of them may be marked as locally defined through
    defineSymbols(). What's left undefined at the end of a statement is either
    turned into an external reference (undefined symbol) or linked to an already
    locally or globally define symbol.

    The methods openBlock(), addStatement() and changeBranch() are considered
    "complete statement points" and cause all the symbols created in the meanwhile
    that didn't pass through defineSymbols() as undefined or otherwise defined
    elsewhere.
    
    For example, in parsing the following code, the noted operations are taken:

    @code
    a = 0           // defineSymbols( a ); addStatement( a = 0 );
    while a \< 5    // openBlock( while a \< 5 );
       a++          // addStatement( a++ );
       if (v=a) > 2 // defineSymbols(v); openBlock( if (v=a) > 2 );
         doThis()   // addStatement(doThis(v));
       elif a < 1   // changeBranch(elif a \< 1);
         doThat()   // addStatement(doThat());
       end          // closeContext(); addStatement(if);
    end             // closeContext(); addStatement(while);
    */
   void checkSymbols();

   /** Adds a statement to the current context.
    \param stmt The statement to be added.
    \see checkSymbols();
    */
   void addStatement( Statement* stmt );

   /** Opens a new block-statement context.
    \param Statement parent The parent that is opening this context.
    \param branch The SynTree under which to add new
    \see checkSymbols();
    */
   void openBlock( Statement* parent, SynTree* branch );

   /** Changes the branch of a block statement context without closing it.
    \param branch The new branch of the topmost statement.

    This is useful when switching branch in swithces or if/else multiple
    block statements without disrupting the block structure.

    \note The parent statement of this block stays the originally pushed statement,
      but bstmt is used to check for undefined symbols, as the
    \see checkSymbols();
    */
   void changeBranch( SynTree* branch);

   /** Opens a new Function statement context.
    \param func The function being created.
    */
   void openFunc( SynFunc *func );
   
   /** Opens a new Class statement context.
    \param cls The class being created.
    \param bIsObject True if this class is declared as singleton object.
    */
   void openClass( Class *cls, bool bIsObject );

   /** Pops the current context.
    When a context is completed, it's onNew*() method is called by this method.
    */
   void closeContext();

   /** Gets the current syntactic tree.
      \return the current syntactic tree where statements are added, or 0 for none.
    */
   SynTree* currentTree() const { return m_st; }

   /** Gets the current statement.
      \return The statement that is parent to this one, or 0 for nothing.
    
      Zero is returned also for topmost-level statements inside functions.
      The returned value is non-zero only if a block statement has been opened.
    */
   Statement* currentStmt() const { return m_cstatement; }

   /** Gets the current function.
      \return The current function or 0 if the parser is parsing in the main code.
    */
   SynFunc* currentFunc() const { return m_cfunc; }

   /** Gets the current class.
      \return The current class, or 0 if the parser is not inside a class.
    */
   Class* currentClass() const { return m_cclass; }

   /** To be called back by rules when the parser state needs to be pushed.
    Rules must call back this method after having pushed the new state in the
    parsers and the new context in this class.

    In this way, as the context is popped, the parser state is automatically
    popped as well.
    */
   void onStatePushed();

   /** Finds a symbol in one of the existing symbol table.
      \param name The name of a symbol to be searched.
      \return A symbol that can be inserted in existing expressions, or
            0 if not found.
    The returned symbol is either a local symbol of the topmost symbol table,
    a closed symbol of intermediate symbol tables or a global symbol in the
    lowest symbol table.

    More precisely, if the symbol is found in the topmost table, it is returned
    as-is, while if it's found in an underlying table, it is returned as-is unless
    it's a LocalSymbol. In that case, a new ClosedSymbol is created and added
    to the topmost table before being returned.
    
    */
   Symbol* findSymbol( const String& name );

private:
   class CCFrame; // forward decl for Context Frames.
   class STFrame; // forward decl fro Symbol table frames.
   
   SourceParser* m_parser;
   
   // Pre-cached syntree for performance.
   SynTree* m_st;

   // Current function, precached for performance.
   Statement* m_cstatement;

   // Current function, precached for performance.
   SynFunc * m_cfunc;

   // Current class, precached for performance.
   Class * m_cclass;

   class Private;
   Private* _p;
};

}

#endif	/* _FALCON_PARSERCONTEXT_H */

/* end of parsercontext.h */
