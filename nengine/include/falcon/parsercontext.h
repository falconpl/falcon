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

 */
class ParserContext {
public:

   /** A frame of the parser context. */
   class CCFrame
   {
      typedef union tag_elem {
         Class* cls;
         SynFunc* func;
         Statement* stmt;
      } t_elem;

      typedef enum tag_type {
         t_none_type,
         t_class_type,
         t_object_type,
         t_func_type,
         t_stmt_type
      } t_type;

      CCFrame();
      CCFrame( Class* cls, bool bIsObject );
      CCFrame( SynFunc* func );
      CCFrame( Statement* stmt, SynTree* st );

   public:
      friend class ParserContext;
      
      /** Syntree element topping this frame. */
      t_elem m_elem;

      /** Type of frame */
      t_type m_type;

      /** Syntree where to add the incoming children */
      SynTree* m_branch;

      /** True if a parser state was pushed at this frame */
      bool m_bStatePushed;

   };

   /** Creates the compiler context.
    \param sp Pointer to the source parser that is using this context.
   */
   ParserContext( SourceParser *sp );
   virtual ~ParserContext();

   /** Called when the input terminates.
    This method is called by the compiler when its input is terminated.

    This might be used by this context to check for context scoping errors
    at the end of file.
    */
   virtual void onInputOver() = 0;

   /** Called back when creating a new variable.
      \param variable The variable that is being created.

    This method is called back when a new global variable is being
    created. Local symbols are managed internally by the context.
    */
   virtual void onNewGlobal( GlobalSymbol* variable ) = 0;

   /** Called back when creating a new function.
      \param function The function that is being created.

    This method is called back when a new fuction is being
    created.

    If the function has also a global name, then onNewGlobal will be called
    \b after onNewFunc; subclasses may cache the function name to understand
    that the new global is referring to this function object.
    */
   virtual void onNewFunc( Function* function ) = 0;

   /** Called back when creating a new class.
      \param cls The variable that is being created.
      \param bIsObj True if the class has been defined as a singleton object

    This method is called back when a new class is being
    created.

    If the function has also a global name, then onNewGlobal will be called
    \b after onNewFunc; subclasses may cache the function name to understand
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


   /** Creates a new variable in the current context.
    \param variable A Variable or symbol name to be created.
    \return A new or already existing symbol.

    This might create a new variable or access an already existing variable
    in the current context.

    \note By default, variables are added also as "extern".
    */
   Symbol* addVariable( const String& variable );

   /** Adds a statement to the current context.
    \param stmt The statement to be added.
    */
   void addStatement( Statement* stmt );

   /** Opens a new block-statement context.
    \param Statement parent The parent that is opening this context.
    \param branch The SynTree under which to add new
    */
   void openBlock( Statement* parent, SynTree* branch );

   /** Changes the branch of a block statement context without closing it.
    \param branch The new branch of the topmost statement.

    This is useful when switching branch in swithces or if/else multiple
    block statements without disrupting the block structure.

    */
   void changeBranch(SynTree* branch);

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

private:
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
