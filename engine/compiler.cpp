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
#include <falcon/deferrorhandler.h>
#include <falcon/stdstreams.h>
#include <falcon/itemid.h>
#include <falcon/fassert.h>

namespace Falcon
{

AliasMap::AliasMap():
   Map( &traits::t_stringptr, &traits::t_voidp )
{
}

//===============================================================
// Compiler
//===============================================================

Compiler::Compiler( Module *mod, Stream* in ):
   m_errhand(0),
   m_module(0),
   m_stream( in ),
   m_constants( &traits::t_string, &traits::t_voidp ),
   m_strict( false ),
   m_language( "C" ),
   m_modVersion( 0 ),
   m_defContext( false ),
   m_delayRaise( false ),
   m_rootError( 0 ),
   m_lexer(0),
   m_root(0),
   m_bParsingFtd(false)
{
   // Initializing now prevents adding predefined constants to the module.
   init();

   m_module = mod;
   m_module->engineVersion( FALCON_VERSION_NUM );

   addPredefs();

   // reset FTD parsing mode
   parsingFtd(false);
}

Compiler::Compiler():
   m_errhand(0),
   m_module( 0 ),
   m_stream( 0 ),
   m_constants( &traits::t_string, &traits::t_voidp ),
   m_strict( false ),
   m_language( "C" ),
   m_modVersion( 0 ),
   m_defContext( false ),
   m_delayRaise( false ),
   m_rootError( 0 ),
   m_lexer(0),
   m_root(0),
   m_lambdaCount(0),
   m_bParsingFtd(false)
{
   reset();
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
   m_lexer = new SrcLexer( this, m_stream );
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

   // reset contants
   MapIterator iter = m_constants.begin();
   while( iter.hasCurrent() ) {
      delete *(Value **) iter.currentValue();
      iter.next();
   }
   m_constants.clear();

   m_alias.clear();
   m_context.clear();
   m_func_ctx.clear();
   m_contextSet.clear();
   m_loops.clear();
   m_statementVals.clear();

   addPredefs();

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
   flc_src_parse( this );
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

      // a correctly compiled module has an entry point starting at 0.
      if ( ! m_root->statements().empty() )
      {
         m_module->entry( 0 );
      }
      return true;
   }

   if ( m_delayRaise && m_rootError != 0 && m_errhand != 0 )
   {
      m_errhand->handleError( m_rootError );
   }

   // eventually prepare for next compilation
   if ( m_rootError != 0 )
   {
      m_rootError->decref();
      m_rootError = 0;
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

   if ( m_errhand != 0 )
   {
      SyntaxError *error = new SyntaxError( ErrorParam(code, line).origin( e_orig_compiler ));
      error->extraDescription( errorp );
      error->module( m_module->path() );

      if ( m_delayRaise )
      {
         if ( m_rootError == 0 )
         {
            m_rootError = new SyntaxError( ErrorParam( e_syntax ).origin( e_orig_compiler ) );
         }
         m_rootError->appendSubError( error );
      }
      else
      {
         m_errhand->handleError( error );
      }

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
      if ( val->asExpr()->type() == Expression::t_array_byte_access )
      {
         raiseError( e_byte_access, lexer()->previousLine() );
         // but proceed
      }
      else {
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
         if ( getFunction() != 0 ) {
            // addlocal must also define the symbol
            sym = addLocalSymbol( val->asSymdef(), false );
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

               sym = addGlobalSymbol( val->asSymdef() );
               sym->declaredAt( lexer()->previousLine() );
               sym->setGlobal();
            }
         }

         val->setSymbol( sym );
      }
      else {
         String *symname = m_module->addString( *staticPrefix() + "#" + *val->asSymdef() );
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


Symbol *Compiler::addLocalSymbol( const String *symname, bool parameter )
{
   // fallback to add global if not in a local table
   FuncDef *func = getFunction();
   if ( func == 0 )
      return addGlobalSymbol( symname );

   SymbolTable &table = func->symtab();
   Symbol *sym = table.findByName( *symname );
   if( sym == 0 )
   {
      if ( m_strict && ! m_defContext )
      {
         raiseError( e_undef_sym, "", lexer()->previousLine() );
      }

      // now we can add the symbol. As we have the string from
      // the module already, we keep it.
      sym = new Symbol( m_module, symname );
      m_module->addSymbol( sym );
      sym->declaredAt( lexer()->previousLine() );
      if ( parameter ) {
         sym = func->addParameter( sym );
      }
      else
      {
         sym = func->addLocal( sym );
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
            fassert( m_functions.end() );
            if ( m_functions.begin() != m_functions.end() )
            {
               const FuncDef *fd_parent = reinterpret_cast<const FuncDef *>
                  ( m_functions.end()->prev()->data() );

               if( fd_parent != 0 )
               {
                  if( fd_parent->symtab().findByName( *val->asSymdef() ) != 0 )
                  {
                     sym = getFunction()->addUndefined(
                        m_module->addSymbol(*val->asSymdef()) );
                  }
               }
            }
         }

         // still nothing?
         if (sym == 0)
         {
            sym = addGlobalSymbol( val->asSymdef() );
         }
         val->setSymbol( sym );
      }
      l->popFront();
   }
   return true;
}


Symbol *Compiler::searchLocalSymbol( const String *symname )
{
   if( m_functions.empty() )
      return searchGlobalSymbol( symname );

   // first search the local symbol aliases
   AliasMap *map = (AliasMap *) m_alias.back();
   Symbol **sympp = (Symbol **) map->find( symname );
   if ( sympp != 0 )
      return *sympp;

   // then try in the local symtab or just return 0.
   FuncDef *fd = (FuncDef *) m_functions.back();
   return fd->symtab().findByName( *symname );
}

Symbol *Compiler::searchOuterSymbol( const String *symname )
{
   ListElement *aliasIter = m_alias.end();
   ListElement *funcIter = m_functions.end();

   while( aliasIter != 0 && funcIter != 0 )
   {
      AliasMap *map = (AliasMap *) aliasIter->data();

      // first search the local symbol aliases
      Symbol **sympp = (Symbol **) map->find( symname );
      if ( sympp != 0 )
         return *sympp;

      // then try in the local symtab or just return 0.
      FuncDef *fd = (FuncDef *) funcIter->data();
      Symbol *sym = fd->symtab().findByName( *symname );
      if ( sym != 0 )
         return sym;

      aliasIter = aliasIter->prev();
      funcIter = funcIter->prev();
   }

   return searchGlobalSymbol( symname );
}


Symbol *Compiler::searchGlobalSymbol( const String *symname )
{
   return module()->findGlobalSymbol( *symname );
}


Symbol *Compiler::addGlobalSymbol( const String *symname )
{
   // is the symbol already defined?
   Symbol *sym = m_module->findGlobalSymbol( *symname );
   if( sym == 0 )
   {
      sym = new Symbol( m_module, symname );
      m_module->addGlobalSymbol( sym );
      sym->declaredAt( lexer()->line() );
   }
   return sym;
}


Symbol *Compiler::addGlobalVar( const String *symname, VarDef *value )
{
   Symbol *sym = addGlobalSymbol( symname );
   sym->declaredAt( lexer()->previousLine() );
   sym->setVar( value );
   return sym;
}


Symbol *Compiler::addAttribute( const String *symname )
{
   // find the global symbol for this.
   Symbol *sym = searchGlobalSymbol( symname );

   // Not defined?
   if( sym == 0 ) {
      sym = addGlobalSymbol( symname );
      sym->declaredAt( lexer()->previousLine() );
   }
   else {
      raiseError( e_already_def,  sym->name() );
   }
   // but change it in an attribute anyhow
   sym->setAttribute();

   return sym;
}


Symbol *Compiler::globalize( const String *symname )
{
   if ( ! isLocalContext() ) {
      // an error should be raised elsewhere.
      return 0;
   }

   // already alaised? raise an error
   AliasMap &map = *(AliasMap *) m_alias.back();
   Symbol **ptr = (Symbol **) map.find( symname );
   if( ptr != 0 )
   {
      raiseError( e_global_again, *symname );
      return *ptr;
   }

   // search for the global symbol that will be aliased
   Symbol *global = m_module->findGlobalSymbol( *symname );
   if ( global == 0 )
   {
      global = m_module->addGlobal( *symname, false );
      global->declaredAt( lexer()->line() );
      // it's defined in the module, the reference will be overwritten with
      // defineVal() -- else it will be searched outside the module.
      // (eventually causing a link error if not found).
   }

   map.insert( symname, global );
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
   Symbol *funcsym = addGlobalSymbol(  addString( cname ) );
   //def->addProperty( addString( "_init" ) , new VarDef( funcsym ) );

   // creates the syntree entry for the symbol; we are using the same line as the class.
   StmtFunction *stmt_ctor = new StmtFunction( cls->line(), funcsym );
   addFunction( stmt_ctor );

   // fills the symbol to be a valid constructor
   FuncDef *fdef = new FuncDef( 0 );
   funcsym->setFunction( fdef );
   def->constructor( funcsym );

   // now we must copy the parameter of the class in the parameters of the constructor.
   MapIterator iter = def->symtab().map().begin();
   GenericVector params( &traits::t_voidp );

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
   addIntConstant( "IntegerType", FLC_ITEM_INT );
   addIntConstant( "NumericType", FLC_ITEM_NUM );
   addIntConstant( "RangeType", FLC_ITEM_RANGE );
   addIntConstant( "AttributeType", FLC_ITEM_ATTRIBUTE );
   addIntConstant( "FunctionType", FLC_ITEM_FUNC );
   addIntConstant( "StringType", FLC_ITEM_STRING );
   addIntConstant( "MemBufType", FLC_ITEM_MEMBUF );
   addIntConstant( "ArrayType", FLC_ITEM_ARRAY );
   addIntConstant( "DictionaryType", FLC_ITEM_DICT );
   addIntConstant( "ObjectType", FLC_ITEM_OBJECT );
   addIntConstant( "ClassType", FLC_ITEM_CLASS );
   addIntConstant( "MethodType", FLC_ITEM_METHOD );
   addIntConstant( "ClassMethodType", FLC_ITEM_CLSMETHOD );
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
   addConstant( name, new Value( m_module->addString( value ) ), line );
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
      // transform them in parameters.
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
            // ok, this must become a parameter...
            sym->setParam();
            sym->itemId( moved );
            moved ++;

            Symbol *parentSym;
            if ( (parentSym = searchLocalSymbol( &sym->name() )) != 0 )
            {
               //... and the parent symbol must be stored in parametric array...
               closureDecl->pushBack( new Value( parentSym ) );
            }
            else {
               // closures can't have undefs.
               raiseError( e_undef_sym, "", sym->declaredAt() );
            }
         }
         else if ( sym->isParam() )
         {
            // push forward all parameters
            sym->itemId( fd->undefined() + sym->itemId() );
         }

         iter.next();
      }

      // no more undefs -- now they are params
      fd->params( fd->params() + fd->undefined() );
      fd->undefined( 0 );

      // ... put it in front of our array and return it.
      closureDecl->pushFront( lambda_call );
      return new Value( closureDecl );
   }
   else
   {
      // just create create a lambda call for this function
      return lambda_call;
   }

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
      VarDef *vd = m_module->addClassProperty( stmt->symbol(), str );
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
      }
   }
}


void Compiler::addEnumerator( const String &str )
{
   Value dummy( (int64) m_enumId );
   addEnumerator( str, &dummy );
}

}

/* end of compiler.cpp */
