/*
   FALCON - The Falcon Programming Language.
   FILE: compiler.cpp

   Core language compiler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 01-08-2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/compiler.h>
#include <falcon/syntree.h>
#include <falcon/src_lexer.h>
#include <falcon/error.h>
#include <falcon/stdstreams.h>
#include <falcon/itemid.h>
#include <falcon/fassert.h>
#include <falcon/path.h>
#include <falcon/intcomp.h>
#include <falcon/modloader.h>
#include <falcon/vm.h>
#include <falcon/stringstream.h>
#include "core_module/core_module.h"

#include <math.h>

namespace Falcon
{

AliasMap::AliasMap():
   Map( &traits::t_stringptr(), &traits::t_voidp() )
{
}

//===============================================================
// Compiler
//===============================================================

Compiler::Compiler( Module *mod, Stream* in ):
   m_constants( &traits::t_string(), &traits::t_voidp() ),
   m_namespaces( &traits::t_string(), &traits::t_voidp() ),
   m_root(0),
   m_errors(0),
   m_optLevel(0),

   m_lexer(0),
   m_stream( in ),

   m_module(0),
   m_enumId(0),

   m_staticPrefix(0),
   m_lambdaCount(0),
   m_closureContexts(0),
   m_tempLine(0),

   m_strict( false ),
   m_language( "C" ),
   m_modVersion( 0 ),
   m_defContext( false ),
   m_bParsingFtd(false),
   m_bInteractive( false ),

   m_rootError( 0 ),
   m_metacomp( 0 ),
   m_serviceVM( 0 ),
   m_serviceLoader( 0 )
{
   init();

   m_module = mod;
   m_module->engineVersion( (FALCON_VERSION_NUM) );

   addPredefs();

   // reset FTD parsing mode
   parsingFtd(false);
}

Compiler::Compiler():
   m_constants( &traits::t_string(), &traits::t_voidp() ),
   m_namespaces( &traits::t_string(), &traits::t_voidp() ),
   m_root(0),
   m_errors(0),
   m_optLevel(0),
   m_lexer(0),
   m_stream( 0 ),
   m_module( 0 ),
   m_enumId( 0 ),
   m_staticPrefix(0),
   m_lambdaCount(0),
   m_closureContexts(0),
   m_tempLine(0),
   m_strict( false ),
   m_language( "C" ),
   m_modVersion( 0 ),
   m_defContext( false ),
   m_bParsingFtd(false),
   m_bInteractive( false ),
   m_rootError( 0 ),
   m_metacomp(0),
   m_serviceVM( 0 ),
   m_serviceLoader( 0 )
{
   addPredefs();
}

Compiler::~Compiler()
{
   // clear doesn't clear the constants
   MapIterator iter = m_constants.begin();
   while( iter.hasCurrent() ) {
      delete *(Value **) iter.currentValue();
      iter.next();
   }

   clear();
}


void Compiler::init()
{
   m_lexer = new SrcLexer( this );
   if ( m_bParsingFtd )
      m_lexer->parsingFtd( true );

   m_root = new SourceTree;

   // context is empty, but the root context set must be placed.
   pushContextSet( &m_root->statements() );

   // and an empty list for the local function undefined values
   List *l = new List;
   m_statementVals.pushBack( l );

   m_errors = 0;
   m_enumId = 0;
   m_staticPrefix = 0;
   m_lambdaCount = 0;
   m_closureContexts = 0;
   m_tempLine = 0;

   // reset the require def status
   m_defContext = false;
}


void Compiler::reset()
{
   if ( m_module != 0 )
   {
      clear();
   }

   m_namespaces.clear();

   m_alias.clear();
   m_context.clear();
   m_func_ctx.clear();
   m_contextSet.clear();
   m_loops.clear();

   m_errors = 0;
   m_enumId = 0;
   m_staticPrefix = 0;
   m_lambdaCount = 0;
   m_closureContexts = 0;
   m_tempLine = 0;

   // reset FTD parsing mode
   parsingFtd(false);
}


void Compiler::clear()
{

   if( m_serviceVM != 0 )
      m_serviceVM->finalize();

   delete m_serviceLoader;
   delete m_metacomp;
   m_serviceVM = 0;
   m_serviceLoader = 0;
   m_metacomp = 0;

   delete m_root;
   delete m_lexer;

   m_root = 0;
   m_lexer= 0;

   m_functions.clear();

   ListElement *valListE = m_statementVals.begin();
   while( valListE != 0 )
   {
      delete (List *) valListE->data();
      valListE = valListE->next();
   }
   m_statementVals.clear();

   if ( m_rootError != 0 )
   {
      m_rootError->decref();
      m_rootError = 0;
   }

   m_module = 0;
   m_stream = 0;
}


bool Compiler::compile( Module *mod, Stream *in )
{
   if ( m_module != 0 )
   {
      clear();
   }

   // m_stream must be configured at init time.
   m_stream = in;
   init();

   m_module = mod;
   m_module->engineVersion( FALCON_VERSION_NUM );

   return compile();
}


bool Compiler::compile()
{
   if ( m_module == 0 || m_stream == 0 )
   {
      raiseError( e_cmp_unprep );
      return false;
   }

   // save directives
   bool bSaveStrict = m_strict;
   String savedLanguage = m_language;
   int64 modver = m_modVersion;

   // parse
   m_lexer->input( m_stream );
   flc_src_parse( this );
   if ( ! m_stream->good() )
   {
      raiseError( new IoError( ErrorParam( e_io_error, __LINE__ )
         .origin( e_orig_compiler )
         .sysError( (uint32) m_stream->lastError() ) ) );
   }
   m_module->language( m_language );
   m_module->version( (uint32) m_modVersion );

   // restore directives
   m_strict = bSaveStrict;
   m_language = savedLanguage;
   m_modVersion = modver;

   // If the context is not empty, then we have something unclosed.
   if ( ! m_context.empty() ) {
      Statement *stmt  = getContext();
      String temp = "from line ";
      temp.writeNumber( (int64) stmt->line() );
      raiseError( e_unclosed_cs, temp );
   }

   // leadout sequence
   if ( m_errors == 0 ) {
      if ( m_root->isExportAll() )
      {
         // export all the symbol in the global module table
         m_module->symbolTable().exportUndefined();
      }

      return true;
   }

   // we had errors.
   return false;
}


void Compiler::raiseError( int code, int line )
{
   raiseError( code, "", line );
}


void Compiler::raiseContextError( int code, int line, int startLine )
{
   if ( line != startLine )
   {
      m_lexer->resetContexts();
      String temp = "from line ";
      temp.writeNumber( (int64) startLine );
      raiseError( code, temp, line );
   }
   else {
      raiseError( code, line );
   }
}


void Compiler::raiseError( int code, const String &errorp, int line )
{
   if ( line == 0 )
      line = lexer()->line()-1;

   SyntaxError *error = new SyntaxError( ErrorParam(code, line)
         .origin( e_orig_compiler ) );
   error->extraDescription( errorp );
   error->module( String(m_module->path()).bufferize() );

   raiseError( error );
}

void Compiler::raiseError( Error *error )
{
   if ( m_rootError == 0 )
   {
      m_rootError = error;
   }
   else
   {
      error->module( m_module->path() );
      m_rootError->appendSubError( error );
      error->decref();
   }

   m_errors++;
}


void Compiler::pushFunction( FuncDef *f )
{
   m_functions.pushBack( f );
   m_statementVals.pushBack( new List );

   m_alias.pushBack( new AliasMap );
}


void Compiler::popFunction()
{
   m_functions.popBack();
   List *l = (List *) m_statementVals.back();
   delete l;
   m_statementVals.popBack();
   DeclarationContext *dc = new DeclarationContext;
   delete dc;
   AliasMap *temp = (AliasMap *) m_alias.back();
   delete temp;
   m_alias.popBack();
}


void Compiler::defineVal( ArrayDecl *val )
{
   ListElement *iter = val->begin();
   while( iter != 0 )
   {
      Value *val = (Value *) iter->data();
      defineVal( val );
      iter = iter->next();
   }
}


void Compiler::defineVal( Value *val )
{
   // raise error for read-only expressions
   if ( val->isExpr() )
   {
      // byte accessors cannot receive values
      if ( val->asExpr()->type() == Expression::t_array_byte_access )
      {
         raiseError( e_byte_access, lexer()->previousLine() );
         // but proceed
      }
      // assignments to accessors and  function returns doesn't define anything.
      else if ( ! (val->asExpr()->type() == Expression::t_obj_access ||
                   val->asExpr()->type() == Expression::t_array_access ||
                   val->asExpr()->type() == Expression::t_funcall )
              )
      {
         Expression *expr = val->asExpr();
         defineVal( expr->first() );
         if( expr->second() != 0 )
         {
            if ( expr->second()->isExpr() &&
                 expr->second()->asExpr()->type() == Expression::t_assign
                 )
            {
               defineVal( expr->second() );
            }
            else if ( expr->isBinaryOperator() )
            {
               raiseError( e_assign_const, lexer()->previousLine() );
            }
         }
      }
   }
   else if ( val->isArray() )
   {
      ListElement *it_s = val->asArray()->begin();
      while( it_s != 0 )
      {
         Value *t = (Value *) it_s->data();
         defineVal( t );
         it_s = it_s->next();
      }
   }
   else if ( val->isImmediate() )
   {
      raiseError( e_assign_const, lexer()->previousLine() );
   }
   else if ( val->isSymdef() )
   {
      if ( staticPrefix() == 0 )
      {
         Symbol *sym;
         // are we in a function that misses that symbol?
         if ( getFunction() != 0 )
         {
            // we're creating the local symbol
            // addlocal must also define the symbol
            sym = addLocalSymbol( *val->asSymdef(), false );
         }
         else {
            // globals symbols that have been added as undefined must stay so.
            sym = m_module->findGlobalSymbol( *val->asSymdef() );
            if ( sym == 0 )
            {
               // values cannot be defined in this way if def is required AND we are not in a def context
               if ( m_strict && ! m_defContext )
               {
                  raiseError( e_undef_sym, "", lexer()->previousLine() );
               }

               sym = addGlobalSymbol( *val->asSymdef() );
               sym->declaredAt( lexer()->previousLine() );
               sym->setGlobal();
            }
         }

         val->setSymbol( sym );
      }
      else {
         String symname = *staticPrefix() + "#" + *val->asSymdef();
         Symbol *gsym = addGlobalSymbol( symname );
         AliasMap &map = *(AliasMap*)m_alias.back();
         map.insert( val->asSymdef(), gsym );
         if( gsym->isUndefined() )
         {
            if ( m_strict && ! m_defContext )
            {
               raiseError( e_undef_sym, "", lexer()->previousLine() - 1);
            }
            gsym->setGlobal();
            gsym->declaredAt( lexer()->previousLine() - 1);
         }
         val->setSymbol( gsym );
      }
   }
}


Symbol *Compiler::addLocalSymbol( const String &symname, bool parameter )
{
   // fallback to add global if not in a local table
   FuncDef *func = getFunction();
   if ( func == 0 )
      return addGlobalSymbol( symname );

   SymbolTable &table = func->symtab();
   Symbol *sym = table.findByName( symname );
   if( sym == 0 )
   {
      // now we can add the symbol. As we have the string from
      // the module already, we keep it.
      sym = new Symbol( m_module, symname );
      m_module->addSymbol( sym );
      sym->declaredAt( lexer()->previousLine() );

      // this flag marks closure forward definitions
      bool taken = false;
      if ( parameter ) {
         sym = func->addParameter( sym );
      }
      else
      {
         // If we're in a closure, we may wish to add
         // a local undefined that will be filled at closure ending
         if ( m_closureContexts && searchLocalSymbol( symname, true ) != 0 )
         {
            taken = true;
            sym = func->addUndefined( sym );
         }
         else
            sym = func->addLocal( sym );
      }

      // in strict mode raise an error if we're not in def, but not if
      // we taken the symbol from some parent.
      if ( !taken && m_strict && !m_defContext )
      {
         raiseError( e_undef_sym, "", lexer()->previousLine() );
      }
   }
   return sym;
}


bool Compiler::checkLocalUndefined()
{
   List *l = (List *) m_statementVals.back();
   while( ! l->empty() )
   {
      Value *val = (Value *) l->front();
      if ( val->isSymdef() )
      {
         Symbol *sym = 0;

         if ( m_closureContexts > 0 )
         {
            // still undefined after some loop?
            // in case of undef == x[undef.len()] we have double
            // un-definition of undef on the same line.
            fassert( m_functions.end() != 0 );
            ListElement* le = m_functions.end();
            const FuncDef *fd_current = reinterpret_cast<const FuncDef *>( le->data() );

            // try to find it locally -- may have been defined in a previous loop
            sym = fd_current->symtab().findByName( *val->asSymdef() );
            if  ( sym == 0 )
            {
               le = le->prev();
               while( le != 0 )
               {
                  const FuncDef *fd_parent = reinterpret_cast<const FuncDef *>( le->data() );

                  if( fd_parent->symtab().findByName( *val->asSymdef() ) != 0 )
                  {
                     sym = m_module->addSymbol(*val->asSymdef());
                     sym->declaredAt( lexer()->previousLine() );
                     getFunction()->addUndefined( sym );
                     break;
                  }

                  le = le->prev();
               }
            }
         }

         // still nothing?
         if (sym == 0)
         {
            // --- import the symbol
            sym = addGlobalSymbol( *val->asSymdef() );
         }

         val->setSymbol( sym );
      }

      l->popFront();
   }
   return true;
}


Symbol *Compiler::searchLocalSymbol( const String &symname, bool recurse )
{
   if( m_functions.empty() )
      return searchGlobalSymbol( symname );

   // first search the local symbol aliases
   AliasMap *map = (AliasMap *) m_alias.back();
   Symbol **sympp = (Symbol **) map->find( &symname );
   if ( sympp != 0 )
      return *sympp;

   // then try in the local symtab or just return 0.
   ListElement* lastFunc = m_functions.end();
   while( lastFunc != 0 )
   {
      FuncDef *fd = (FuncDef *)lastFunc->data();
      Symbol *found;
      if( (found = fd->symtab().findByName( symname )) )
         return found;

      if ( ! recurse )
         return 0;

      lastFunc = lastFunc->prev();
   }

   // definitely not found.
   return 0;
}

Symbol *Compiler::searchOuterSymbol( const String &symname )
{
   ListElement *aliasIter = m_alias.end();
   ListElement *funcIter = m_functions.end();

   while( aliasIter != 0 && funcIter != 0 )
   {
      AliasMap *map = (AliasMap *) aliasIter->data();

      // first search the local symbol aliases
      Symbol **sympp = (Symbol **) map->find( &symname );
      if ( sympp != 0 )
         return *sympp;

      // then try in the local symtab or just return 0.
      FuncDef *fd = (FuncDef *) funcIter->data();
      Symbol *sym = fd->symtab().findByName( symname );
      if ( sym != 0 )
         return sym;

      aliasIter = aliasIter->prev();
      funcIter = funcIter->prev();
   }

   return searchGlobalSymbol( symname );
}


Symbol *Compiler::searchGlobalSymbol( const String &symname )
{
   return module()->findGlobalSymbol( symname );
}


Symbol *Compiler::addGlobalSymbol( const String &symname )
{
   // is the symbol already defined?
   Symbol *sym = m_module->findGlobalSymbol( symname );
   bool imported = false;
   if( sym == 0 )
   {
      // check if it is authorized.
      // Unauthorized symbols are namespaced symbol not declared in import all clauses
      uint32 dotpos;
      if( (dotpos = symname.rfind ( "." ) ) != String::npos )
      {
         // Namespaced symbol
         String nspace = symname.subString( 0, dotpos );
         // change self into our name

         void **mode = (void **) m_namespaces.find( &nspace );

         // if it's not a namespace, then it's an hard-built symbol as i.e. class._init
         if ( mode != 0 )
         {
            // we wouldn't have a namespaced symbol if the lexer didn't find it was already in
            if( *mode == 0 )
            {
               // if we were authorized, the symbol would have been created by
               // the clauses itself.
               raiseError( e_undef_sym, symname );
               // but add it anyhow.
            }

            // namespaced symbols are always imported
            imported = true;
         }
      }

      sym = new Symbol( m_module, symname );
      m_module->addGlobalSymbol( sym );
      sym->declaredAt( lexer()->previousLine() );
      sym->imported( imported );
   }
   return sym;
}


Symbol *Compiler::addGlobalVar( const String &symname, VarDef *value )
{
   Symbol *sym = addGlobalSymbol( symname );
   sym->declaredAt( lexer()->previousLine() );
   sym->setVar( value );
   return sym;
}

Symbol *Compiler::globalize( const String &symname )
{
   if ( ! isLocalContext() ) {
      // an error should be raised elsewhere.
      return 0;
   }

   // already alaised? raise an error
   AliasMap &map = *(AliasMap *) m_alias.back();
   Symbol **ptr = (Symbol **) map.find( &symname );
   if( ptr != 0 )
   {
      raiseError( e_global_again, symname );
      return *ptr;
   }

   // search for the global symbol that will be aliased
   Symbol *global = m_module->findGlobalSymbol( symname );
   if ( global == 0 )
   {
      global = m_module->addGlobal( symname, false );
      global->declaredAt( lexer()->line() );
      // it's defined in the module, the reference will be overwritten with
      // defineVal() -- else it will be searched outside the module.
      // (eventually causing a link error if not found).
   }

   map.insert( &symname, global );
   return global;
}


StmtFunction *Compiler::buildCtorFor( StmtClass *cls )
{
   Symbol *sym = cls->symbol();

   // Make sure we are not overdoing this.
   fassert( sym->isClass() );
   fassert( sym->getClassDef()->constructor() == 0 );

   // creates a name for the constructor
   ClassDef *def = sym->getClassDef();
   String cname = sym->name() + "._init";

   // creates an empty symbol
   Symbol *funcsym = addGlobalSymbol( cname );
   //def->addProperty( addString( "_init" ) , new VarDef( funcsym ) );

   // creates the syntree entry for the symbol; we are using the same line as the class.
   StmtFunction *stmt_ctor = new StmtFunction( cls->line(), funcsym );
   addFunction( stmt_ctor );

   // fills the symbol to be a valid constructor
   FuncDef *fdef = new FuncDef( 0, 0 );
   funcsym->setFunction( fdef );
   def->constructor( funcsym );

   // now we must copy the parameter of the class in the parameters of the constructor.
   MapIterator iter = def->symtab().map().begin();
   GenericVector params( &traits::t_voidp() );

   while( iter.hasCurrent() )
   {
      Symbol *symptr = *(Symbol **) iter.currentValue();
      if ( symptr->isParam() )
      {
         Symbol *p = m_module->addSymbol( symptr->name() );
         fdef->addParameter( p );
         p->itemId( symptr->itemId() );
      }
      iter.next();
   }

   cls->ctorFunction( stmt_ctor );
   stmt_ctor->setConstructorFor( cls );

   return stmt_ctor;
}


void Compiler::addPredefs()
{

   addIntConstant( "NilType", FLC_ITEM_NIL );
   addIntConstant( "BooleanType", FLC_ITEM_BOOL );
   //addIntConstant( "IntegerType", FLC_ITEM_INT );
   addIntConstant( "NumericType", FLC_ITEM_NUM );
   addIntConstant( "RangeType", FLC_ITEM_RANGE );
   addIntConstant( "FunctionType", FLC_ITEM_FUNC );
   addIntConstant( "StringType", FLC_ITEM_STRING );
   addIntConstant( "LBindType", FLC_ITEM_LBIND );
   addIntConstant( "MemBufType", FLC_ITEM_MEMBUF );
   addIntConstant( "ArrayType", FLC_ITEM_ARRAY );
   addIntConstant( "DictionaryType", FLC_ITEM_DICT );
   addIntConstant( "ObjectType", FLC_ITEM_OBJECT );
   addIntConstant( "ClassType", FLC_ITEM_CLASS );
   addIntConstant( "MethodType", FLC_ITEM_METHOD );
   addIntConstant( "ClassMethodType", FLC_ITEM_CLSMETHOD );
   addIntConstant( "UnboundType", FLC_ITEM_UNB );

}


void Compiler::addAttribute( const String &name, Value *val, uint32 line )
{
   String n = name;
   FuncDef* fd = getFunction();
   if( fd == 0 )
   {
      m_module->addAttribute( name, val->genVarDef() );
   }
   else {
      fd->addAttrib(name, val->genVarDef() );
   }
}

void Compiler::addIntConstant( const String &name, int64 value, uint32 line )
{
   addConstant( name, new Value( value ), line );
}

void Compiler::addNilConstant( const String &name, uint32 line )
{
   addConstant( name, new Value(), line );
}

void Compiler::addStringConstant( const String &name, const String &value, uint32 line )
{
   if ( m_module != 0 )
      addConstant( name, new Value( m_module->addString( value ) ), line );
   else {
      // we'll leak, but oh, well...
      //TODO: fix the leak
      addConstant( name, new Value( new String( value ) ), line );
   }

}

void Compiler::addNumConstant( const String &name, numeric value, uint32 line )
{
   addConstant( name, new Value( value ), line );
}

void Compiler::addConstant( const String &name, Value *val, uint32 line )
{
   // is a constant with the same name defined?
   if ( m_constants.find( &name ) != 0 ) {
      raiseError( e_already_def, name, line );
      return;
   }

   // is a symbol with the same name defined ?
   // Module may be zero (i.e. for pre-defined Falcon constants)
   if ( m_module != 0 && m_module->findGlobalSymbol( name ) != 0 ) {
      raiseError( e_already_def, name, line );
      return;
   }

   if( ! val->isImmediate() ) {
      raiseError( e_assign_const, name, line );
      return;
   }

   // create a global const symbol
   /* It's a thing I must think about
   Symbol *sym = new Symbol( m_module->addString( name ) );
   sym->setConst( val->genVarDef() );
   m_module->addGlobalSymbol( sym );
   */

   // add the constant to the compiler.
   String temp( name );
   m_constants.insert( &temp, val );
}


void Compiler::closeFunction()
{
   StmtFunction *func = static_cast<Falcon::StmtFunction *>( getContext() );
   Symbol *fsym = func->symbol();
   FuncDef *def = fsym->getFuncDef();

   // has this function a static block?
   if ( func->hasStatic() )
   {
      def->onceItemId( m_module->addGlobal( "_once_" + fsym->name(), false )->itemId() );
   }

   popContext();
   popFunctionContext();
   popContextSet();
   popFunction();
}


bool Compiler::parsingFtd() const
{
   return m_bParsingFtd;
}


void Compiler::parsingFtd( bool b )
{
   m_bParsingFtd = b;

   if ( m_lexer != 0 )
      m_lexer->parsingFtd( b );
}


bool Compiler::setDirective( const String &directive, const String &value, bool bRaise )
{
   bool bWrongVal = false;

   if ( directive == "strict" )
   {
      if ( value == "on" )
      {
         m_strict = true;
         return true;
      }
      else if ( value == "off" )
      {
         m_strict = false;
         return true;
      }

      bWrongVal = true;
   }
   else if ( directive == "lang" )
   {
      m_language = value;
      return true;
   }

   // ...
   // if we're here we have either a wrong directive or a wrong value.
   if ( bRaise )
   {
      if ( bWrongVal )
         raiseError( e_directive_value, directive + "=" + value, m_lexer->line() );
      else
         raiseError( e_directive_unk, directive, m_lexer->line() );
   }

   return true;
}


bool Compiler::setDirective( const String &directive, int64 value, bool bRaise )
{
   bool bWrongVal = false;

   if ( directive == "strict" || directive == "lang" )
   {
      bWrongVal = true;
   }
   else if ( directive == "version" )
   {
      m_modVersion = value;
      return true;
   }

   // if we're here we have either a wrong directive or a wrong value.
   if ( bRaise )
   {
      if ( bWrongVal )
      {
         String temp = directive;
         temp += "=";
         temp.writeNumber( value );
         raiseError( e_directive_value, temp, m_lexer->line() );
      }
      else
         raiseError( e_directive_unk, directive, m_lexer->line() );
   }

   return true;
}

Value *Compiler::closeClosure()
{
   // first, close it as a normal function, but without mangling with the
   // local function table.
   StmtFunction *func = static_cast<Falcon::StmtFunction *>( getContext() );
   FuncDef *fd = func->symbol()->getFuncDef();

   // has this function a static block?
   if ( func->hasStatic() )
   {
      fd->onceItemId( m_module->addGlobal( "_once_" + func->symbol()->name(), false )->itemId() );
   }

   popContext();
   popFunctionContext();
   popContextSet();
   popFunction();
   decClosureContext();

   // we're going to need a lambda call
   Value *lambda_call = new Value( new Falcon::Expression( Expression::t_lambda , new Value( func->symbol() ) ) );

   // Is there some undefined?
   if( fd->undefined() > 0 )
   {
      // we have to find all the local variables that exist in the upper context and
      // transform them in the first part of the local array.
      SymbolTable &funcTable = fd->symtab();
      ArrayDecl *closureDecl = new ArrayDecl;

      //First; put parameters away, so that we can reorder them.
      const Map &symbols = funcTable.map();
      MapIterator iter = symbols.begin();
      int moved = 0;

      while( iter.hasCurrent() )
      {
         Symbol *sym = *(Symbol **) iter.currentValue();
         if ( sym->isLocalUndef() )
         {
            // ok, this must become a local symbol...
            sym->setLocal();
            sym->itemId( moved );
            moved ++;

            // and now let's find the father's having this local undefined.
            fassert( m_functions.end() != 0 );
            ListElement* le = m_functions.end();
            Symbol *parentSym = 0;

            while( le != 0 )
            {
               FuncDef *fd_parent = ( FuncDef *) le->data();

               // when we find it...
               if( ( parentSym = fd_parent->symtab().findByName( sym->name() )) != 0 )
               {
                  // we must now force the declaration of local undefined for this
                  // symbol up to this closure, so that it sent down the stack on runtime.
                  le = le->next();
                  while( le != 0 )
                  {
                     fd_parent = (FuncDef *) le->data();
                     parentSym = fd_parent->addUndefined( m_module->addSymbol( sym->name() ) );
                     le = le->next();
                  }

                  //... and the parent symbol must be stored in parametric array...
                  closureDecl->pushBack( new Value( parentSym ) );
                  break;
               }

               le = le->prev();
            }

            if ( parentSym == 0 )
            {
               // closures can't have undefs.
               raiseError( e_undef_sym, "", sym->declaredAt() );
            }
         }
         else if ( sym->isLocal() )
         {
            // push forward all the locals
            sym->itemId( fd->undefined() + sym->itemId() );
         }

         iter.next();
      }

      // no more undefs -- now they are locals
      fd->locals( fd->locals() + fd->undefined() );
      fd->undefined( 0 );

      // ... put as third element of the lambda call.
      lambda_call->asExpr()->second( new Value( closureDecl ) );
   }

   // just create create a lambda call for this function
   return lambda_call;
}


void Compiler::addEnumerator( const String &str, Value *val )
{
   StmtClass *stmt = static_cast< StmtClass *>( getContext() );
   ClassDef *cd = stmt->symbol()->getClassDef();

   if ( cd->hasProperty( str ) )
   {
      raiseError( e_already_def, str, lexer()->previousLine() );
   }
   else
   {
      VarDef *vd = &m_module->addClassProperty( stmt->symbol(), str );
      switch( val->type() )
      {
         case Value::t_nil :
            // nothing to do
            break;

         case Value::t_imm_integer:
            vd->setInteger( val->asInteger() );
            m_enumId = val->asInteger() + 1;
            break;

         case Value::t_imm_num:
            vd->setNumeric( val->asNumeric() );
            m_enumId = int( val->asNumeric() ) + 1;
            break;

         case Value::t_imm_string:
            vd->setString( m_module->addString( *val->asString() ) );
            break;

         case Value::t_imm_bool:
            vd->setBool( val->asBool() );
            break;

         default:
            break;
      }
   }
}


void Compiler::addEnumerator( const String &str )
{
   Value dummy( (int64) m_enumId );
   addEnumerator( str, &dummy );
}


bool Compiler::isNamespace( const String &symName )
{
   return m_namespaces.find( &symName ) != 0;
}


void Compiler::addNamespace( const String &nspace, const String &alias, bool full, bool isFilename )
{
   // Purge special namespace names
   String nselfed;

   if ( alias.size() == 0 )
   {
      // if the namespace starts with self, add also the namespace
      // with the same name of the module
      if( isFilename )
      {
         // we got to convert "/" into "."
         Path fconv( nspace );
         nselfed = fconv.getLocation() + "." + fconv.getFile();
         uint32 pos = 0;

         if ( nselfed.getCharAt(0) == '/' )
            nselfed = nselfed.subString( 1 );

         while( (pos = nselfed.find( "/", pos ) ) != String::npos ) {
            nselfed.setCharAt( pos, '.' );
         }
      }
      else if ( nspace.getCharAt(0) == '.' ) {
         nselfed = nspace.subString( 1 );
      }
      else if ( nspace.find( "self." ) == 0 ) {
         nselfed = nspace.subString(5);
      }
      else {
         nselfed = nspace;
      }
   }
   else {
         nselfed = alias;
   }

   // Already added?
   void **res = (void **) m_namespaces.find( &nselfed );

   if ( res == 0 )
   {
      // yes? -- add it
      m_namespaces.insert( &nselfed, full ? (void *)1 : (void *) 0 );

      // we have to insert in the namespaces all the sub-namespaces.
      uint32 dotpos = nselfed.find( "." );
      while( dotpos != String::npos )
      {
         String subSpace = nselfed.subString( 0, dotpos );

         // don't overwrite in case we already imported something from it.
         void *oldNs = m_namespaces.find( &subSpace );
         if ( oldNs == 0 )
         {
            // we set a default of 0 for them as we normally import nothing.
            m_namespaces.insert( &subSpace, 0 );
         }

         dotpos = nselfed.find( ".", dotpos+1 );
      }

      m_module->addDepend( nselfed, nspace, true, isFilename );
   }
   // no?
   else {
      // -- eventually change to load all.
      if ( *res == 0 && full )
         *res = (void *) 1;

      // -- and check if this is a mis-redefinition
      // raise an error if you try to alias a different module with an alredy existing namespace
      ModuleDepData* mdPrev = m_module->dependencies().findModule( nselfed );
      if ( mdPrev != 0 && mdPrev->moduleName() != nspace )
      {
         raiseError( e_ns_clash, nselfed, lexer()->previousLine() );
      }
   }
}


Symbol *Compiler::importAlias( const String &symName, const String &fromMod, const String &alias, bool filename )
{
   // add the dependency
   m_module->addDepend( fromMod, true, filename );

   // add the alias
   Falcon::Symbol *sym = new Symbol( m_module, alias );
   m_module->addGlobalSymbol( sym );
   sym->declaredAt( lexer()->previousLine() );

   // sets the alias
   sym->setImportAlias( symName, fromMod, filename );

   return sym;
}


void Compiler::importSymbols( List *lst, const String& pre, const String& alias, bool isFilename )
{
   String prefix = pre;

   // add the namespace if not previously known
   if ( prefix.size() != 0 )
   {
      if( alias.size() != 0 )
      {
         addNamespace( prefix, alias, false, isFilename );
         prefix = alias;
      }
      else {
         addNamespace( prefix, "", false, isFilename );
      }
   }

   String fprefix;
   if( isFilename && alias.size() == 0 && prefix.size() != 0 )
   {
      // we got to convert "/" into "."
      Path fconv( prefix );
      fprefix = fconv.getLocation() + "." + fconv.getFile();

      uint32 pos = 0;
      if ( fprefix.getCharAt(0) == '/' )
         fprefix = fprefix.subString(1);

      while( (pos = fprefix.find( "/", pos ) ) != String::npos ) {
         fprefix.setCharAt( pos, '.' );
      }

   }

   Falcon::ListElement *li = lst->begin();

   while( li != 0 ) {
      Falcon::String& symName = *(String *) li->data();

      if( prefix.size() != 0 )
      {
         if( isFilename && alias.size() == 0 ) {
            symName = fprefix + "." + symName;
         }
         else if ( prefix.getCharAt(0) == '.' ) {
            symName = prefix.subString( 1 ) + "." + symName;
         }
         else if ( prefix.find( "self." ) == 0 ) {
            symName = prefix.subString(5) + "." + symName;
         }
         else
            symName = prefix + "." + symName;
      }


      Falcon::Symbol *sym = new Symbol( m_module, symName );
      m_module->addGlobalSymbol( sym );
      sym->declaredAt( lexer()->previousLine() );
      sym->imported(true);

      delete &symName;

      li = li->next();
   }
   delete lst;
}

void Compiler::metaCompile( const String &data, int startline )
{
   String ioEnc, srcEnc;
   Engine::getEncodings( srcEnc, ioEnc );

   // evantually turn on the meta-compiler
   if ( m_metacomp == 0 )
   {
      if ( m_serviceVM == 0 )
      {
         m_serviceVM = new VMachine;
         m_serviceVM->appSearchPath( searchPath() );
         Module*cm = core_module_init();
         m_serviceVM->link( cm );
         cm->decref();
      }

      if ( m_serviceLoader == 0 )
      {
         m_serviceLoader = new ModuleLoader( searchPath() );
         m_serviceLoader->sourceEncoding( srcEnc );
      }

      m_metacomp = new InteractiveCompiler( m_serviceLoader, m_serviceVM );
      m_metacomp->searchPath( searchPath() );

      // not incremental...
      m_metacomp->lexer()->incremental( false );
      // do not confuse our user...
      m_metacomp->module()->path( module()->path() );
      m_metacomp->module()->name( "[meta] " + module()->name() );

      // transfer the constants.
      MapIterator iter = m_constants.begin();
      while( iter.hasCurrent() ) {
         if ( m_metacomp->m_constants.find( iter.currentKey() ) == 0 )
         {
            m_metacomp->addConstant(
               *(String *)iter.currentKey(),
               new Value( **(Value **) iter.currentValue()) );
         }
         iter.next();
      }

   }

   StringStream* output = new StringStream;
   // set the same encoding as the source that we're parsing.
   m_serviceVM->stdOut( TranscoderFactory( srcEnc, output, true ) );


   // set current line in lexer of the meta compiler
   m_metacomp->tempLine( startline );
   try
   {
      m_metacomp->compileAll( data );

      // something has been written
      if ( output->length() != 0 )
      {
         // pass it to the lexer
         output->seekBegin(0);
         StringStream *transferred = new StringStream();
         transferred->transfer( *output );
         m_lexer->appendStream( transferred );
      }
   }
   catch( Error *e )
   {
      raiseError( e );
   }
}

}

/* end of compiler.cpp */
