/*
   FALCON - The Falcon Programming Language.
   FILE: compiler.h

   Main Falcon source compiler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom giu 6 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_COMPILER_H
#define FALCON_COMPILER_H

#include <falcon/syntree.h>
#include <falcon/error.h>
#include <falcon/common.h>
#include <falcon/string.h>
#include <falcon/module.h>
#include <falcon/genericlist.h>
#include <falcon/basealloc.h>

extern "C" int flc_src_parse( void *param );

namespace Falcon
{

class SrcLexer;
class Stream;
class InteractiveCompiler;
class VMachine;
class ModuleLoader;

/**
   ( const String *, Symbol * )
*/
class FALCON_DYN_CLASS AliasMap: public Map
{
public:

   AliasMap();
};

class FALCON_DYN_CLASS DeclarationContext: public BaseAlloc
{
   byte m_value;
public:
   DeclarationContext(){ m_value = 0; }
   DeclarationContext( const DeclarationContext &dc ) {
      m_value = dc.m_value;
   }

   DeclarationContext &operator =( const DeclarationContext &other ) {
      m_value = other.m_value; return *this;
   }

   DeclarationContext &setGlobalQuery() { m_value = 0; return *this; }
   DeclarationContext &setQuery() { m_value &= ~0x1; return *this; }
   DeclarationContext &setAssign() { m_value |= 0x1; return *this; }
   DeclarationContext &setGlobalBody() { m_value &= ~(0x2|0x4|0x8); return *this; }
   DeclarationContext &setFunctionBody() { m_value |= 0x2; m_value &= ~(0x4); return *this; }
   DeclarationContext &setClassBody() { m_value |= 0x4; m_value &= ~(0x2); return *this; }
   DeclarationContext &setStatic() { m_value |= 0x8; return *this; }
   DeclarationContext &setNonStatic() { m_value &= ~0x8; return *this; }

};


/** FALCON source compiler.
   This class is responsible for creating a syntactic tree
   given a linear input stream. The stream may be from file, standard
   input or from a memory buffer ( the compiler does
   not need the ability of random file access, so even a network
   stream may be used ).
*/
class FALCON_DYN_CLASS Compiler: public BaseAlloc
{
protected:
   /** Declaration context.
      Depending on where a variable is initialized, or how a symbol is declared,
      the final symbol added to the module may be different. This enumeration is
      used to keep track of the current declaration context.
   */
   typedef enum {
      e_dc_global,
      e_dc_param,
      e_dc_static_func,
      e_dc_func_body,
      e_dc_static_class,
      e_dc_class_body
   } t_decl_context;

   /** Map of constants.
      (String &, Value *)
   */
   Map m_constants;

   /** Map of namespaces.
      (String &, void)
   */
   Map m_namespaces;

   SourceTree *m_root;

   int m_errors;
   int m_optLevel;

   SrcLexer *m_lexer;
   Stream *m_stream;

   /** This is the module that is being formed in the meanwhile. */
   Module *m_module;

   int64 m_enumId;

   /** Leading instruction that owns currently parsed statements */
   List m_context;
   /** Context limited to leading function instructions. */
   List m_func_ctx;

   /** Area to save currently parsed statements. */
   List m_contextSet;
   /** Leading instruction, specialized for loops. */
   List m_loops;

   /** Last statement's symbols.
       The type of symbols cannot be determined by the context while they are built; they get
       defined as the compiler understands the surrounding context. However, it is an error to
       reference undefined symbols in local context (i.e. lambdas, functions etc.).
       This list has a reference to each symbol that has been built during the last statement
       parsing. At the end of the statement, the list is scanned for undefined symbol, and
       an error is risen in case any is found.
       list of Value *
   */
   List m_statementVals;

   /** Stack of currently active functions.
       Can be nested in case of i.e. lambas.
       (FuncDef *)
   */
   List m_functions;

   /** Aliased symbols are stored here.
      List of alias maps.
   */
   List m_alias;

   /** The static prefix is the name of the symbol currently declaring the static namespace.
      Do not delete: we're not owners.
   */
   const String *m_staticPrefix;

   int m_lambdaCount;
   int m_closureContexts;
   int m_tempLine;

   /** Directive strict. */
   bool m_strict;

   /** Directive language. */
   String m_language;

   /** Directive version. */
   int64 m_modVersion;

   bool m_defContext;

   bool m_bParsingFtd;

   bool m_bInteractive;

   Error *m_rootError;
   InteractiveCompiler *m_metacomp;
   VMachine *m_serviceVM;
   ModuleLoader *m_serviceLoader;

   /** Search path inherited from upper facilities. */
   String m_searchPath;

   /** Removes all the structures and temporary data used to compile a file.
      This function is called automatically by the various compile() and
      destructors.
   */
   void clear();

   /** Initializes structures and variables used for compilation.
      This function is called automatically by the various compile() and
      destructors.
   */
   void init();

   /** Add predefined symbols and constants.
      This method prepares the compiler so that it has basic constant symbols
      set and in place; embeding apps may wish to provide different base
      symbols.
   */
  void addPredefs();

public:
   /** Creates an empty compiler.
      This constructor doesn't set a stream and a module for the compiler.
      It is intended for repeated usage through compile( Module *, Stream *).
   */
   Compiler();

   /** Creates the compiler setting a default module and input stream.
      This configures this instance as a single-file-compilation only compiler.
      After the compile() call, the instance may (should) be disposed.
      However, after calling this constructor it is possible to use the
      compiler( Module *, Stram *) as well.
   */
   Compiler( Module *mod, Stream *input );
   /** Destroys the compiler.
      Internally calls clear()
   */
   virtual ~Compiler();

   /** Reset compiler settings to defaults and prepares for a new compilation.
      Precisely, this function:
      # destroys tree and function information from previous run, if they exist.
      # clears the constants and fills them with the Falcon language default constants
      # clears the ftd compilation flag.

      This function should be called before a repeated compilation; then the caller is
      free to add specific application constants and setting, and finally call the
      compile( Module *, Stream *) method.

      Directives are automatically cleared at the end of a compilation, and they keep the value they
      had before. This allows to set directives from outside and have scripts locally modify their
      directives.
   */
   void reset();

   /** Compiles the module given in the constructor.
      This method is to be used for one-time only compilation (build the compiler, compile,
      destroy the compiler), when the Compiler( Module *, Stream *) constructor version
      has been used.

      Otherwise, it will raise an error and exit.
   */
   bool compile();

   /** Compile a module from a stream.
      This version of the function is suitable to be used multiple times for the same compiler.
      The caller should call resetDefaults(), give the compiler the wished setings and then
      call compiler.
      \param mod a newly allocated and empty module that will be filled by the compilation
      \param input the stream from which to read the source
      \return false on compilation failed.
   */
   bool compile( Module *mod, Stream *input );

   /** Front-end to raiseError( *e ).

      The compiler doesn't throw the error list until the compilation is over.
      error raisal is delayed until the end of the compilation step.
   */
   void raiseError( int errorNum, int errorLine=0);

   /** Raises an error.

      The compiler doesn't throw the error list until the compilation is over.
      error raisal is delayed until the end of the compilation step.
   */
   void raiseError( Error *e );

   /** Raises an error related to a context problem.
      The error reports the line where the problem has been detected, and the line
      that begun current faulty context.
      \param code the error code.
      \param line the line where the error is detected
      \param startLine initial line of the context.
   */
   void raiseContextError( int code, int line, int startLine );
   void raiseError( int errorNum, const String &errorp, int errorLine=0);
   void addError() { m_errors++; }

   /** Searches a symbol in the local context.
    *
    *  If the current context is the global context, then it just calls
    *  searchGlobalSymbol. If there is a local context, the symbol is serched
    *  in the current context; in case it's not found, this method returns
    *  if bRecurse is false.
    *
    *  If bRecurse is true, the symbol is searched down in the parent contexts
    *  until found or until there is no more local context. In that case, the
    *  method returns 0; so if there is the need to bind a local symbol or a global
    *  one if local symbols are not found, this must be done by the caller
    *  calling searchGlobalSymbol when this method returns 0.
    *
    *  \param symname The symbol name as a pointer to a module string.
    *  \param bRecurse true to bind symbols from any parent, false to search in the local context.
    *  \return The symbol if found or 0 otherwise.
    */
   Symbol *searchLocalSymbol( const String &symname, bool bRecurse = false );
   Symbol *searchGlobalSymbol( const String &symname );
   Symbol *addLocalSymbol( const String &symname, bool parameter );
   Symbol *addGlobalSymbol( const String &symname );
   Symbol *searchOuterSymbol( const String &symname );

   /** Creates a symbol that will be an initially defined global variable.
      The global variables may be created with an initial values (i.e. for
      static declarations). This function adds the global symbol for the
      variable and sets it to the default value.
   */
   Symbol *addGlobalVar( const String &symname, VarDef *value );
   bool isLocalContext() { return ! m_functions.empty(); }

   /** Seek a constant in the predefined constant list.
      If the constant is found, the function returns the value associated with the given constant.
      Constant values are owned by the compiler (yet the constant strings are still held in the
      module), and are destroyed at compiler destruction.

      \note just a placeholder for now
      \param name the constant to be searched
      \return the value of the constant or 0 if the constant doesn't exists.
   */
   const Value *getConstant( const String &name ) {
      Value **findp = (Value **) m_constants.find( &name );
      if ( findp != 0 )
         return *findp;
      return 0;
   }

   /** Adds a direct load request to the module being compiled.
      \param name The name of the module to be loaded
      \param isFilename True if the name to be loaded is actually a filename path.
   */
   virtual void addLoad( const String &name, bool isFilename ) {
      m_module->addDepend( name, false, isFilename );
   }

   // Inlines
   void addStatement( Statement *stm ) { if ( stm != 0 ) getContextSet()->push_back( stm ); }
   void addFunction( Statement *stm ) { if ( stm != 0 ) m_root->functions().push_back( stm ); }
   void addClass( Statement *stm ) { if ( stm != 0 ) m_root->classes().push_back( stm ); }
   void pushLoop( Statement *stm ) { m_loops.pushBack( stm ); }
   void pushFunction( FuncDef *f );
   void pushContext( Statement *stm ) { m_context.pushBack( stm ); }
   void pushContextSet( StatementList *list ) { m_contextSet.pushBack( list ); }
   Statement *getContext() const { if ( m_context.empty() ) return 0; return (Statement *) m_context.back(); }
   Statement *getLoop() const { if ( m_loops.empty() ) return 0; return (Statement *) m_loops.back(); }
   StatementList *getContextSet() const { if ( m_contextSet.empty() ) return 0; return (StatementList *)m_contextSet.back(); }
   FuncDef * getFunction() const { if ( m_functions.empty() ) return 0; return (FuncDef *) m_functions.back(); }
   void popLoop() { m_loops.popBack(); }
   void popContext() { m_context.popBack(); }
   void popContextSet() { m_contextSet.popBack(); }
   void popFunction();
   void pushFunctionContext( StmtFunction *func ) { m_func_ctx.pushBack( func ); }
   void popFunctionContext() { if ( !m_func_ctx.empty() ) m_func_ctx.popBack(); }
   StmtFunction *getFunctionContext() const { if ( m_func_ctx.empty() ) return 0; return (StmtFunction*) m_func_ctx.back(); }

   String *addString( const String &str ) { return m_module->addString( str ); }

   SrcLexer *lexer() const { return m_lexer; }
   Stream *stream() const { return m_stream; }

   int lambdaCount() const { return m_lambdaCount; }
   void incLambdaCount() { m_lambdaCount++; }

   void addNilConstant( const String &name, uint32 line=0 );
   void addIntConstant( const String &name, int64 value, uint32 line=0 );
   void addNumConstant( const String &name, numeric value, uint32 line=0 );
   void addStringConstant( const String &name, const String &value, uint32 line=0 );
   void addConstant( const String &name, Value *val, uint32 line=0 );

   /** Adds an attribute to the currently active context-sensible symbol */
   void addAttribute( const String &name, Value *val, uint32 line=0 );


   SourceTree *sourceTree() const { return m_root; }
   Module *module() const { return m_module; }

   int errors() const { return m_errors; }
   /** Process an include instruction.
      In Falcon, \b include is a compile time instruction more than a directive.
      Falcon does not support pre-processing or directives by design. The \b include
      statement takes as argument a single immediate string or an expression that
      can be statically evaluated into a string (i.e. consts or string sums), and
      includes it by changing the internal file stream. This causes an as-is
      inclusion at lexer level.
   */
   //void include( const char *filename );
   //void includePath( const Hstring &incp );

   /** Instruct the compiler that this value is used as a definition.
      This is used to turn local undefined into local variables, or
      global undefinded into globals. Variables may rightfully become
      something else later on (i.e. functions) however.

      @param val The value to be inspected in search for defined symbols.
   */
   void defineVal( Value *val );

   /** Define all the values int the given array definition.
      As the array definition is to the left of an assignment,
      all the atomic symbols that are found in the array definition
      are to be defined. Of course, non atomic symbols (as functions,
      other array defintions and so on) are NOT to be defined.

      @param val The array of left-assigment values, possibly symbols
   */
   void defineVal( ArrayDecl *val );

   Symbol  *globalize( const String &str );

   /** Checks if the current statemsnt has referenced a locally undefined symbol. */
   bool checkLocalUndefined();
   void addSymdef( Value *val ) { List *l = (List *) m_statementVals.back(); l->pushBack( val ); }

   /** Return the current static prefix, if any.
      Zero shall be returned if the current symbol is not currently using the static prefix.
   */
   const String *staticPrefix() const { return m_staticPrefix; }
   void staticPrefix( const String *v ) { m_staticPrefix = v; }

   /** Builds the constructor function for a given class.
      This is an utility that creates a ._init suffixed function statement
      and symbol; the symbol is added as the constructor for the class stored
      in the parameter, while the StmtFunction object is inserted in the
      functions syntax tree and returned.
      \param sym a symbol containing a ClassDef for which a constructor function must be built.
      \return A newly created StmtFunction object to hold the source tree for the constructor.
   */
   StmtFunction *buildCtorFor( StmtClass *sym );

   /** Cleanup for function closing. */
   void closeFunction();

   /** Store current line for later error signaling. */
   void tempLine( int line ) { m_tempLine = line; }
   int tempLine() const { return m_tempLine; }

   /** Activate "strict" feature.
      When turned on, the compiler will raise an undefined symbol when assigning this values
      outside a "def" statement.
   */
   void strictMode( bool breq ) { m_strict = breq; }
   bool strictMode() const { return m_strict; }

   /** Are we parsing a normal file or an escaped template file? */
   bool parsingFtd() const;
   void parsingFtd( bool b );

   /** Set directive as string value.
      In case the directive doesn't exist or doesnt accept the given value as valid,
      an error may be raised. Applications setting directives externally may
      give bRaise false to prevent error raising and manage internally directive set
      failure.
      \param directive the name of the directive to be set.
      \param value the value that the given directive should be given.
      \param bRaise true in case of invalid directive or value, also raise an error
      \return true on success, false on failure
   */
   bool setDirective( const String &directive, const String &value, bool bRaise = true );

   /** Set directive as string value.
      In case the directive doesn't exist or doesnt accept the given value as valid,
      an error may be raised. Applications setting directives externally may
      give bRaise false to prevent error raising and manage internally directive set
      failure.
      \param directive the name of the directive to be set.
      \param value the value that the given directive should be given.
      \param bRaise true in case of invalid directive or value, also raise an error
      \return true on success, false on failure
   */
   bool setDirective( const String &directive, int64 value, bool bRaise = true );

   void defContext( bool ctx ) { m_defContext = ctx; }
   bool defContext() const { return m_defContext; }

   /** Closes the currently worked on closure */
   Value *closeClosure();
   void incClosureContext() { m_closureContexts++; }
   void decClosureContext() { m_closureContexts--; }

   /** Add an enumeration item to current enumeration. */
   void addEnumerator( const String &name, Value *value );

   /** Add an enumeration item to current enumeration.
      This version assigns the enumerated value a progressive integer.
   */
   void addEnumerator( const String &name );

   void resetEnum() { m_enumId = 0; }

   /** Return true if the current symbol is actually a namespace.
      Checks if sym was previosly declared as a namespace with
      addNamspace().
      \note Namespaces can contain dots.
      \param symName the name that may be possibly a namespace.
   */
   bool isNamespace( const String &symName );

   /** Adds a known namespace.
      \param nspace The namespace to be added (as module name).
      \param alias If not empty, will be the alias under which the module will be locally known.
      \param full If true, import all symbols.
      \param filename If true, the load request was for a direct filename.
   */
   virtual void addNamespace( const String &nspace, const String &alias, bool full=false, bool filename=false );

   /** Import the symbols named in a List.
      The \b lst parameter contains a list of String*
      which will be imported. If a prefix is given, then
      the prefix is added to the list of known namespaces,
      and it is added in front of the symbol to be imported.

      This will force the VM to search the symbol only in
      the specified namespace (that is, in the module named
      after the prefix).

      The function is meant to be used in the conjunction
      with the parser, and will destroy both the string
      in \b lst and \b lst itself.
      \param lst The list of symbols (disposeable strings) to be loaded
      \param prefix The module name or path
      \param alias The namespace. If not given, will be the name of the module.
      \param filename if true the module load request is for a direct file name
   */
   virtual void importSymbols( List *lst, const String &prefix=String(), const String &alias=String(), bool filename=false );

   /** Import a single aliased symbol from a module.
      This tries to resolve the given symbol in the given symName in the target from module
      during link time, and assings the local alias to it
      \param symName The symbol name to be aliased.
      \param fromMod The module name or path.
      \param alias The namespace. If not given, will be the name of the module.
      \param filename if true the module load request is for a direct file name.
      \param return The newly added symbol.
   */
   virtual Symbol* importAlias( const String &symName, const String &fromMod, const String &alias, bool filename=false );

   /** Performs a meta compilation.
      If any data is written on the metacompiler output stream, the
      stream is immediately sent to the lexer for further compilation.
   */
   void metaCompile( const String &data, int startline );

   /** Gets the service VM for this compiler.
      Used to create the associate meta-compiler.
      The VM is owned by this compiler (or of its meta-compiler).
   */
   VMachine *serviceVM() const { return m_serviceVM; }

   /** Gets the service VM for this compiler.
      Used to create the associate meta-compiler.
      The loader is owned by this compiler (or of its meta-compiler).
   */
   ModuleLoader *serviceLoader() const { return m_serviceLoader; }

   /** Sets the service VM for this compiler.
      Used to create the associate meta-compiler.
      The VM is owned by this compiler (or of its meta-compiler).

      If a metacompilation is required, the ownership of this VM is transferred
      to the meta compiler. If a service VM is not set and a meta-compilation
      is requried, the compiler creates a VM on the fly.
   */
   void serviceVM( VMachine *vm ) { m_serviceVM = vm; }

   /** Sets the service VM for this compiler.
      Used to create the associate meta-compiler.
      The loader is owned by this compiler (or of its meta-compiler).

      If a metacompilation is required, the ownership of this loader is transferred
      to the meta compiler. If a service loader is not set and a meta-compilation
      is requried, the compiler creates a loader on the fly.
   */
   void serviceLoader(ModuleLoader *l) { m_serviceLoader = l; }

   /** Passes the ownership of the error structure to the caller.

      This allows the caller to get the errors received during the processing of the
      last compilation and handle them separately.

      The ownership is passed to the caller, or in other world, the list of errors
      in this compiler is zeroed and the reference count of the returned error
      list is not changed.

      \return 0 if the compiler was not in error state, or a list of one or more errors if it
      raised some errors during the processing of the files.
   */
   Error *detachErrors() { Error *e = m_rootError; m_rootError = 0; m_errors = 0; return e; }

   /** Sets this compiler as interactive.
      A compiler meant to run code from the command line has different
      rules; for example, it can accept autoexpressions without raising
      a "statement does nothing" error, as the meaning of expressions
      on the command line is just that to be evaluated and their result
      being interactively displayed.
   */
   void setInteractive( bool bint ) { m_bInteractive = bint; }

   /** Checks if this compiler is as an interactive I/O */
   bool isInteractive() const { return m_bInteractive; }

   /** Return the search path inherited from upper facilities.
      This search path is used to drive module loading in macro compiler.
   */
   const String& searchPath() const { return m_searchPath; }

   /** Sets the compiler-specific search path.
      This search path is used to drive module loading in macro compiler.
   */
   void searchPath( const String& path ) { m_searchPath = path; }
};

} // end of namespace

#endif

/* end of compiler.h */
