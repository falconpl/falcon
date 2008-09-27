/*
   FALCON - The Falcon Programming Language.
   FILE: compiler.cpp

   Assembly compiler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2005-08-22

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>
#include <fasm/comp.h>
#include <falcon/common.h>
#include <fasm/clexer.h>
#include <fasm/pseudo.h>
#include <falcon/string.h>
#include <falcon/types.h>
#include <falcon/pcodes.h>
#include <falcon/module.h>
#include <falcon/stream.h>
#include <falcon/fassert.h>
#include <falcon/error.h>


/** \file Falcon assembly compiler main file. */

namespace Falcon
{

Pseudo *AsmCompiler::regA_Inst() { return m_lexer->regA_Inst(); }
Pseudo *AsmCompiler::regB_Inst() { return m_lexer->regB_Inst(); }
Pseudo *AsmCompiler::regS1_Inst() { return m_lexer->regS1_Inst(); }
Pseudo *AsmCompiler::regS2_Inst() { return m_lexer->regS2_Inst(); }
Pseudo *AsmCompiler::regL1_Inst() { return m_lexer->regL1_Inst(); }
Pseudo *AsmCompiler::regL2_Inst() { return m_lexer->regL2_Inst(); }
Pseudo *AsmCompiler::nil_Inst() { return m_lexer->nil_Inst(); }

AsmCompiler::AsmCompiler( Module *mod, Stream *in ):
      m_labels( &traits::t_stringptr(), &traits::t_voidp() ),
      m_current( 0 ),
      m_switchEntriesInt( &traits::t_pseudoptr(), &traits::t_pseudoptr() ),
      m_switchEntriesRng( &traits::t_pseudoptr(), &traits::t_pseudoptr() ),
      m_switchEntriesStr( &traits::t_pseudoptr(), &traits::t_pseudoptr() ),
      m_switchEntriesObj( &traits::t_pseudoptr(), &traits::t_pseudoptr() ),
      m_errhand(0),
      m_module( mod ),
      m_outTemp( new StringStream ),
      m_currentLine( 1 ),
      m_pc(0)
{
   m_lexer = new AsmLexer( mod, this, in );
   m_errors = 0;
   m_switchItem = 0;
   m_switchJump = 0;
   m_switchEntryNil = 0;
   m_module->engineVersion( FALCON_VERSION_NUM );

   //m_switchObjList.deletor( Pseudo_Deletor );
}


AsmCompiler::~AsmCompiler()
{
   MapIterator iter = m_labels.begin();
   while( iter.hasCurrent() )
   {
      LabelDef *def = *(LabelDef**) iter.currentValue();
      delete def;
      iter.next();
   }

   clearSwitch();

   delete m_lexer;
   delete m_outTemp;
}

bool AsmCompiler::compile()
{
   fasm_parse( this );

   // verify for missing labels
   MapIterator loc_iter = m_labels.begin();
   while( loc_iter.hasCurrent() ) {
      LabelDef *def = *(LabelDef**) loc_iter.currentValue();
      if ( ! def->defined() ) {
         const String *name = *(const String **) loc_iter.currentKey();
         raiseError( e_undef_label, *name );
      }
      loc_iter.next();
   }

   return (m_errors == 0);
}

bool AsmCompiler::defined( const String &name ) const
{
   void *data = m_labels.find( &name );
   if ( data != 0 )
      return true;

   Symbol *sym;
   if ( name.getCharAt( 0 ) == '*' )
      sym = m_module->findGlobalSymbol( name.subString( 1 ) );
   else
      sym = m_module->findGlobalSymbol( name );

   if ( sym != 0 )
      return true;

   return false;
}

void AsmCompiler::raiseError( int code, int line )
{
   raiseError( code, "", line );
}

void AsmCompiler::raiseError( int code, const String &errorp, int line )
{
   int character = 0;
   if ( line == 0 ) {
      line = lexer()->line();
      character = lexer()->character();
   }

   if ( m_errhand != 0 )
   {
      SyntaxError *error = new SyntaxError(
         ErrorParam( code, line ).extra( errorp ).origin( e_orig_assembler ).
         module( m_module->name() ).
         chr( character )
      );


      m_errhand->handleError( error );
      error->decref();
   }

   m_errors++;
}

void AsmCompiler::setModuleName( Pseudo *val )
{
   m_module->name( val->asString() );
   delete val;
}


const String *AsmCompiler::addString( const String &data )
{
   return m_module->addString( data );
}


void AsmCompiler::addDLine( Pseudo *line )
{
   m_currentLine = (uint32) line->asInt();
   m_module->addLineInfo( m_pc + (uint32) m_outTemp->tell(), static_cast<uint32>(line->asInt()) );
   delete line;
}

void AsmCompiler::addEntry()
{
   // not used anymore
}

void AsmCompiler::addGlobal( Pseudo *val, Pseudo *line, bool exp )
{
   if ( defined( val->asString() ) )
   {
      raiseError( e_already_def, val->asString() );
   }
   else {
      m_module->addGlobal( val->asString(), exp )->declaredAt( (int32) line->asInt() );
   }
   delete line;
   delete val;
}


void AsmCompiler::addVar( Pseudo *psym, Pseudo *pval, Pseudo *line, bool exp )
{
   if ( defined( psym->asString() ) )
   {
      raiseError( e_already_def, psym->asString() );
   }
   else {
      Symbol *sym = m_module->addGlobal( psym->asString() );
      sym->declaredAt( (int32) line->asInt() );
      VarDef *vd;
      switch( pval->type() )
      {
         case Pseudo::imm_int: vd = new VarDef( VarDef::t_int, pval->asInt() ); break;
         case Pseudo::imm_double: vd = new VarDef( pval->asDouble() ); break;
         case Pseudo::imm_string: vd = new VarDef( m_module->addString( pval->asString() ) ); break;
         default:
            vd = new VarDef();
      }

      sym->setVar( vd );
      sym->exported( exp );
   }
   delete line;
   delete psym;
   if ( pval->disposeable() )
      delete pval;
}

void AsmCompiler::addConst( Pseudo *psym, Pseudo *pval, bool exp )
{
   if ( defined( psym->asString() ) )
   {
      raiseError( e_already_def, psym->asString() );
   }
   else {
      Symbol *sym = m_module->addGlobal( psym->asString() );
      VarDef *vd;
      switch( pval->type() )
      {
         case Pseudo::imm_int: vd = new VarDef( VarDef::t_int, pval->asInt() ); break;
         case Pseudo::imm_double: vd = new VarDef( pval->asDouble() ); break;
         case Pseudo::imm_string: vd = new VarDef( m_module->addString( pval->asString() ) ); break;
         default:
            vd = new VarDef();
      }

      sym->setConst( vd );
      sym->exported( exp );
   }
   delete psym;
   if ( pval->disposeable() )
      delete pval;
}

void AsmCompiler::addAttrib( Pseudo *psym, Pseudo *line, bool exp )
{
   if ( defined( psym->asString() ) )
   {
      raiseError( e_already_def, psym->asString() );
   }
   else {
      Symbol *sym = m_module->addGlobal( psym->asString() );
      sym->declaredAt( (int32) line->asInt() );

      sym->setAttribute();
      sym->exported( exp );
   }
   delete psym;
   delete line;
}


void AsmCompiler::addExport( Pseudo *val )
{
   Symbol *sym = m_module->findGlobalSymbol( val->asString() );
   if ( sym == 0 )
   {
      raiseError( e_export_undef, val->asString() );
   }
   else {
      sym->exported( true );
   }
   delete val;
}

void AsmCompiler::addLoad( Pseudo *val, bool isFile )
{
   m_module->addDepend( val->asString(), false, isFile );
   delete val;
}

void AsmCompiler::addImport( Pseudo *val, Pseudo *line, Pseudo *mod, Pseudo *alias, bool isFile )
{

   String symname = val->asString();

   // prefix the name
   if ( alias != 0 )
      symname = alias->asString() + "." + symname;
   else if ( mod != 0 )
      symname = mod->asString() + "." + symname;

   if ( defined( symname ) )
   {
      raiseError( e_already_def, symname );
   }
   else
   {
      Symbol *sym = m_module->addSymbol( symname );
      // sym is undef (extern) by default.
      m_module->addGlobalSymbol( sym )->declaredAt( (int32) line->asInt() );
      // specify explicit import
      sym->imported(true);
   }

   // add the module dependency, if required
   if( mod != 0 )
   {
      if ( alias != 0 )
         m_module->addDepend( alias->asString(), mod->asString(), true, isFile ); // private
      else
         m_module->addDepend( mod->asString(), true, isFile ); // private
   }

   delete val;
   delete line;
   delete mod;
   delete alias;
}

void AsmCompiler::addLocal( Pseudo *val, Pseudo *line )
{
   if ( m_current == 0 || ! m_current->isFunction() )
   {
      raiseError( e_no_local );
   }
   else {
      FuncDef *func = m_current->getFuncDef();

      if ( func->symtab().findByName( val->asString() ) )
      {
         raiseError( e_already_def, val->asString() );
      }
      else {
         if ( func->locals() == 65530 )
            raiseError( e_too_locals  );
         Symbol *sym = m_module->addSymbol( val->asString() );
         sym->declaredAt( (int32) line->asInt() );
         func->addLocal( sym );
      }
   }

   delete val;
   delete line;
}

void AsmCompiler::addParam( Pseudo *val, Pseudo *line )
{
   if ( m_current == 0  )
   {
      raiseError( e_no_local );
   }
   else if( m_current->isFunction() )
   {
      FuncDef *func = m_current->getFuncDef();

      if ( func->symtab().findByName( val->asString() ) )
      {
         raiseError( e_already_def, val->asString() );
      }
      else {
         if ( func->params() == 65530 )
            raiseError( e_too_locals  );
         Symbol *sym = m_module->addSymbol( val->asString() );
         sym->declaredAt( (int32) line->asInt() );
         func->addParameter( sym );
      }
   }
   else if( m_current->isClass() )  {
      ClassDef *cd = m_current->getClassDef();
      if ( cd->symtab().findByName( val->asString() ) )
      {
         raiseError( e_already_def, val->asString() );
      }
      else {
         if ( cd->params() == 65530 )
            raiseError( e_too_locals  );
         Symbol *sym = m_module->addSymbol( val->asString() );
         sym->declaredAt( (int32) line->asInt() );
         cd->addParameter( sym );
      }
   }
   else {
      raiseError( e_no_local );
   }

   delete val;
   delete line;
}

void AsmCompiler::classHas( Pseudo *val )
{
   if ( m_current == 0 ||  ! m_current->isClass() )
   {
      raiseError( e_no_class );
   }
   else {
      ClassDef *def = m_current->getClassDef();
      def->has().pushBack( val->asSymbol() );
   }

   delete val;
}

void AsmCompiler::classHasnt( Pseudo *val )
{
   if ( m_current == 0 ||  ! m_current->isClass() )
   {
      raiseError( e_no_class );
   }
   else {
      ClassDef *def = m_current->getClassDef();
      def->hasnt().pushBack( val->asSymbol() );
   }

   delete val;
}

void AsmCompiler::addProperty( Pseudo *val, Pseudo *defval  )
{
   if ( m_current == 0 || ! m_current->isClass() )
   {
      raiseError( e_no_class );
   }
   else {
      ClassDef *def = m_current->getClassDef();
      VarDef *vd;
      switch( defval->type() ) {
         case Pseudo::imm_int: vd = new VarDef( VarDef::t_int, defval->asInt() ); break;
         case Pseudo::imm_double: vd = new VarDef( defval->asDouble() ); break;
         case Pseudo::imm_string: vd = new VarDef( m_module->addString( defval->asString() ) ); break;
         case Pseudo::tsymbol:
         {
            // symbols that can be used as class property initializers must
            // be defined at global module scope.
            Symbol *sym = m_module->symbolTable().findByName( defval->asSymbol()->name() );
            if ( sym == 0 ) {
               vd = new VarDef();
               raiseError( e_undef_sym, defval->asSymbol()->name() );
            }
            else {
               vd = new VarDef( sym );
            }
         }
         break;
         default:
            vd = new VarDef();
      }

      String *propname = m_module->addString( val->asString() );

      if ( def->hasProperty( *propname ) )
      {
         raiseError( e_prop_adef, *propname );
         delete vd;
      }
      else {
         def->addProperty( propname, vd );
         if ( def->properties().size() == 0xFFFF )
            raiseError( e_too_props );
      }
   }

   delete val;
   if ( defval->disposeable() )
      delete defval;
}


void AsmCompiler::addClassCtor( Pseudo *ctor )
{
   if ( m_current == 0 || ! m_current->isClass() )
   {
      raiseError( e_no_class );
   }
   else {
      m_current->getClassDef()->constructor( ctor->asSymbol() );
   }

   delete ctor;
}


void AsmCompiler::addPropRef( Pseudo *val, Pseudo *defval )
{
   if ( m_current == 0 || ! m_current->isClass() )
   {
      raiseError( e_no_class );
   }
   else {
      ClassDef *def = m_current->getClassDef();
      // this function can be called only with symbols in defval
      fassert( defval->type() == Pseudo::tsymbol );

      // symbols that can be used as class property initializers must
      // be defined at global module scope.
      Symbol *sym = defval->asSymbol();
      String *propname = m_module->addString( val->asString() );

      if ( def->hasProperty( *propname ) )
      {
         raiseError( e_prop_adef, *propname);
      }
      else {
         VarDef *vd = new VarDef( VarDef::t_reference, sym );
         def->addProperty( propname, vd );
         if ( def->properties().size() == 0xFFFF )
            raiseError( e_too_props );
      }
   }

   delete val;
   delete defval;
}


void AsmCompiler::addFrom( Pseudo *val )
{
   if ( m_current == 0 || ! m_current->isClass() )
   {
      raiseError( e_no_class );
   }
   // Seeking for a property, as from and properties share the same namespace
   else {
      ClassDef *def = m_current->getClassDef();
      Symbol *base_class = m_module->symbolTable().findByName( val->asString() );
      if ( base_class == 0 ) {
         raiseError( e_undef_sym, val->asString() );
      }
      else {
         InheritDef *in_def = new InheritDef( base_class );

         if ( ! def->addInheritance( in_def ) )
         {
            raiseError( e_from_adef, val->asString() );
            delete in_def;
         }
         else {
            if ( def->inheritance().size() == 255 )
               raiseError( e_too_froms );
         }
      }
   }

   delete val;
}


void AsmCompiler::addExtern( Pseudo *val, Pseudo *line )
{
   if ( defined( val->asString() ) )
   {
      raiseError( e_already_def, val->asString() );
   }
   else {
      Symbol *sym = m_module->addSymbol( val->asString() );
      // sym is undef (extern) by default.
      m_module->addGlobalSymbol( sym )->declaredAt( (int32) line->asInt() );
      // don't add imported; that is for import
   }
   delete line;
   delete val;
}


void AsmCompiler::addFunction( Pseudo *val, Pseudo *line, bool exp )
{
   if ( defined( val->asString() ) )
   {
      raiseError( e_already_def, val->asString()  );
   }
   else {
      Symbol *sym = m_module->addSymbol( val->asString() );
      // function is not yet defined.
      sym->setFunction( new FuncDef( 0, 0 ) );
      sym->exported( exp );
      m_module->addGlobalSymbol( sym )->declaredAt( (int32) line->asInt() );
   }

   delete line;
   delete val;
}

void AsmCompiler::addClass( Pseudo *val, Pseudo *line, bool exp )
{

   if ( defined( val->asString() ) )
   {
      raiseError( e_already_def, val->asString() );
   }
   else {
      Symbol *sym = m_module->addSymbol( val->asString() );
      // function is not yet defined.
      sym->setClass( new ClassDef );
      sym->exported( exp );
      m_module->addGlobalSymbol( sym )->declaredAt( (int32) line->asInt() );
   }

   delete line;
   delete val;
}

void AsmCompiler::addFuncDef( Pseudo *val, bool exp )
{
   // close the main symbol
   closeMain();

   // see if there is a forward reference.
   Symbol *sym = m_module->findGlobalSymbol( val->asString() );
   if ( sym != 0 )
   {
      if ( ! sym->isFunction() ) {
         raiseError( e_already_def, val->asString() );
      }
      else {
         FuncDef *fd = sym->getFuncDef();
         if ( exp )
            sym->exported( true );

         if ( fd->codeSize() != 0 ) {
            raiseError( e_already_def, val->asString() );
         }
      }
   }

   // defined it as a brand new thing
   else {
      // pseudo strings for symbols are already in module
      sym = m_module->addSymbol( val->asString() );
      sym->setFunction( new FuncDef( 0, 0 ) );
      sym->exported( exp );
      m_module->addGlobalSymbol( sym );
   }

   m_current = sym;
   sym->getFuncDef()->basePC( m_pc );

   delete val;
}


void AsmCompiler::addClassDef( Pseudo *val, bool exp )
{
   // close the main symbol
   closeMain();

   // see if there is a forward reference.
   Symbol *sym = m_module->findGlobalSymbol( val->asString() );
   if ( sym != 0 )
   {
      if ( ! sym->isClass() ) {
         raiseError( e_already_def, val->asString() );
      }
      else {
         if ( exp )
            sym->exported( true );

         if ( sym->getClassDef()->basePC() != 0x0 ) {
            raiseError( e_already_def, val->asString() );
         }
         // just a marker, actually is not used for classes.
         sym->getClassDef()->basePC( 0x1 );
      }
   }
   // defined it as a brand new thing
   else {
      // pseudo strings for symbols are already defined in the module.
      sym = m_module->addSymbol( val->asString() );
      sym->setClass( new ClassDef );
      sym->exported( exp );
      m_module->addGlobalSymbol( sym );
      sym->declaredAt( lexer()->line() - 1 );
   }

   m_current = sym;

   delete val;
}


void AsmCompiler::closeMain()
{
   uint32 codeSize = m_outTemp->length();
   if ( codeSize > 0 && m_pc == 0 )
   {
      Symbol *sym = m_module->findGlobalSymbol( "__main__" );
      if ( sym != 0 )
      {
         raiseError( e_already_def, "__main__" );
      }
      else {
         // pseudo strings for symbols are already defined in the module.
         sym = m_module->addSymbol( *m_module->addString("__main__") );
         sym->setFunction( new FuncDef( m_outTemp->closeToBuffer(), codeSize ) );
         m_pc += codeSize;
         sym->exported( false );
         m_module->addGlobalSymbol( sym );
         delete m_outTemp;
         m_outTemp = new StringStream;
      }
   }
}

void AsmCompiler::addInherit( Pseudo *baseclass )
{
   // we must be in a class context
   if ( m_current == 0 || ! m_current->isClass() )
   {
      raiseError( e_no_class );
   }
   else {
      // the symbol must be already defined.
      Symbol *cls = baseclass->asSymbol();
      ClassDef *cd = m_current->getClassDef();
      InheritDef *inherit = new InheritDef( cls );
      cd->addInheritance( inherit );
   }
   delete baseclass;
}


void AsmCompiler::addInheritParam( Pseudo *param )
{
   // we must be in a class context
   if ( m_current == 0 || ! m_current->isClass() )
   {
      raiseError( e_no_class );
   }
   else {
      // the symbol must be already defined.
      ClassDef *cd = m_current->getClassDef();
      if ( cd->inheritance().empty() ) {
         fassert( false ); // impossible if the program is correct
      }
      else {
         InheritDef *inherit = ( InheritDef *) cd->inheritance().back();
         VarDef *vd;
         switch( param->type() )
         {
            case Pseudo::tnil: vd = new VarDef(); break;
            case Pseudo::imm_int: vd = new VarDef( param->asInt() ); break;
            case Pseudo::imm_double: vd = new VarDef( param->asDouble() ); break;
            case Pseudo::imm_string: vd = new VarDef( m_module->addString( param->asString() ) ); break;
            case Pseudo::tsymbol: vd = new VarDef( param->asSymbol() ); break;
            default: fassert( false ); // imposible if the program is correct
         }

         inherit->addParameter( vd );
      }
   }

   if ( param->disposeable() )
      delete param;
}



void AsmCompiler::addFuncEnd()
{
   if ( m_current == 0 )
      raiseError( e_end_no_loc );
   else {
      m_pc += m_outTemp->length();
      m_current->getFuncDef()->codeSize( m_outTemp->length() );
      m_current->getFuncDef()->code( m_outTemp->closeToBuffer() );
      m_pc += m_current->getFuncDef()->codeSize();
      delete m_outTemp;
      m_outTemp = new StringStream;
      m_current = 0;
   }
}

void AsmCompiler::addStatic()
{
   if ( m_current == 0 )
   {
      raiseError( e_static_notin_func );
      return;
   }

   FuncDef *def = m_current->getFuncDef();
   if ( def->onceItemId() == FuncDef::NO_STATE )
   {
      Symbol *glob = m_module->addGlobal( "_once_" + m_current->name(), false );
      // do we have it already (usually, this is the case in correctly built asm files.
      if ( glob == 0 )
      {
         glob = m_module->symbolTable().findByName( "_once_" + m_current->name() );
         fassert( glob != 0 );
      }
      def->onceItemId( glob->itemId() );
   }
   // as more than one ONCE is possible in assembly, we should not raise an error.
}

LabelDef *AsmCompiler::addLabel( const String &name )
{
   LabelDef **defp = (LabelDef **) m_labels.find( &name );
   if ( defp == 0 ) {
      LabelDef *created = new LabelDef( name );
      m_labels.insert( &created->name(), created );
      return created;
   }
   return *defp;
}

void AsmCompiler::defineLabel( LabelDef *def )
{
   if ( def->defined() )
   {
      raiseError( e_already_def, def->name() );
   }
   else {
      def->defineNow( m_outTemp );
   }
}

void AsmCompiler::addInstance( Pseudo *cls, Pseudo *object, Pseudo *line, bool exp )
{
   if ( defined( object->asString() ) )
   {
      raiseError( e_already_def, object->asString()  );
   }
   else {
      Symbol *sym = m_module->addSymbol( object->asString() );
      // function is not yet defined.
      sym->setInstance( cls->asSymbol() );
      sym->exported( exp );
      m_module->addGlobalSymbol( sym )->declaredAt( (int32) line->asInt() );;
   }

   delete line;
   delete object;
   delete cls;
}

unsigned char AsmCompiler::paramDesc( Pseudo *op1 ) const
{
   if ( op1 == 0 )
      return P_PARAM_NOTUSED;

   if( op1->fixed() )
   {
      switch( op1->type() )
      {
         case Pseudo::imm_int:
         case Pseudo::imm_string:
         case Pseudo::tsymbol:
         case Pseudo::tname:
            return P_PARAM_NTD32;
         case Pseudo::imm_double: return P_PARAM_NTD64;
         default:
            break;
      }
   }
   else {
      switch( op1->type() )
      {
         case Pseudo::imm_true: return P_PARAM_TRUE;
         case Pseudo::imm_false: return P_PARAM_FALSE;
         case Pseudo::imm_int: return P_PARAM_INT64;
         case Pseudo::imm_double: return P_PARAM_NUM;
         case Pseudo::imm_string: return P_PARAM_STRID;
         case Pseudo::tlbind: return P_PARAM_LBIND;
         case Pseudo::tsymbol:
            if ( isLocal( op1 ) ) return P_PARAM_LOCID;
            if ( isParam( op1 ) ) return P_PARAM_PARID;
            return P_PARAM_GLOBID;
         case Pseudo::tname:
             return P_PARAM_INT32;
         case Pseudo::tregA: return P_PARAM_REGA;
         case Pseudo::tregB: return P_PARAM_REGB;
         case Pseudo::tregS1: return P_PARAM_REGS1;
         case Pseudo::tregS2: return P_PARAM_REGS2;
         case Pseudo::tregL1: return P_PARAM_REGL1;
         case Pseudo::tregL2: return P_PARAM_REGL2;
         case Pseudo::tnil: return P_PARAM_NIL;
         default:
            break;
      }
   }

   return 0;
}


void AsmCompiler::clearSwitch()
{
   if ( m_switchItem != 0 && m_switchItem->disposeable() )
      delete m_switchItem;
   m_switchItem = 0;

   if ( m_switchEntryNil != 0 && m_switchEntryNil->disposeable() )
      delete m_switchEntryNil;
   m_switchEntryNil = 0;

   m_switchEntriesInt.clear();
   m_switchEntriesRng.clear();
   m_switchEntriesStr.clear();
   m_switchEntriesObj.clear();
   m_switchObjList.clear();
}

void AsmCompiler::addDSwitch( Pseudo *val, Pseudo *jmp_end, bool bselect )
{
   if ( m_switchItem == 0 )
   {
      m_switchIsSelect = bselect;
      m_switchItem = val;
      m_switchJump = jmp_end;
   }
   else {
      raiseError( e_switch_again );
      if (val->disposeable() )
         delete val;
      if ( jmp_end->disposeable() )
         delete jmp_end;
   }
}

void AsmCompiler::addDCase( Pseudo *val, Pseudo *jump, Pseudo *second )
{
   // see if we have a range entry
   if ( second != 0 )
   {
      Pseudo *temp = new Pseudo( val->line(), (int32) val->asInt(), (int32) second->asInt() );

      if ( val->disposeable() )
         delete val;
      if ( second->disposeable() )
         delete second;
      val = temp;
   }

   if ( m_switchItem != 0 )
   {
      Map *switchEntries;

      // depending on the type of the case, select the correct map
      switch( val->type() )
      {
         case Pseudo::tnil:
            if ( m_switchEntryNil != 0 )
               raiseError( e_dup_case );

            m_switchEntryNil = jump;

            if ( val->disposeable() )
               delete val;
            val = 0;
         break;

         case Pseudo::imm_int: switchEntries = &m_switchEntriesInt; break;
         case Pseudo::imm_range: switchEntries = &m_switchEntriesRng; break;
         case Pseudo::imm_string: switchEntries = &m_switchEntriesStr; break;
         case Pseudo::tsymbol:
            // mark this as a pseudo needing symbol ID rather than item ID.
            val->fixed( true );
            switchEntries = &m_switchEntriesObj;
            m_switchObjList.pushBack( val );
         break;
            
         default:
            break;
      }

      // are we filling something different than the NIL entry?
      if ( val != 0 )
      {
         Pseudo **value = (Pseudo **) switchEntries->find( val );
         if( value != 0 ) {
            raiseError( e_dup_case );
            if ( val->disposeable() )
               delete val;
         }
         else
            switchEntries->insert( val, jump );
      }
   }
   else {
      raiseError( e_switch_case );
   }

}

void AsmCompiler::addDEndSwitch()
{

   if ( m_switchItem != 0 )
   {
      // prepare the opcode
      byte opcode = m_switchIsSelect ? P_SELE : P_SWCH;
      m_outTemp->write( reinterpret_cast<char *>(&opcode), 1 );
      opcode =  P_PARAM_NTD32;
      m_outTemp->write( reinterpret_cast<char *>(&opcode), 1 );
      opcode = paramDesc( m_switchItem ) ;
      m_outTemp->write( reinterpret_cast<char *>(&opcode), 1 );
      opcode = P_PARAM_NTD64;
      m_outTemp->write( reinterpret_cast<char *>(&opcode), 1 );

      // prepare end of switch position
      m_switchJump->write( m_outTemp );

      //write the item
      m_switchItem->write( m_outTemp );

      // compose and write the item table size
      int64 sizeInt = (int16) m_switchEntriesInt.size();
      int64 sizeRng = (int16) m_switchEntriesRng.size();
      int64 sizeStr = (int16) m_switchEntriesStr.size();
      int64 sizeObj = (int16) m_switchEntriesObj.size();

      int64 sizes = endianInt64( sizeInt << 48 | sizeRng << 32 | sizeStr << 16 | sizeObj );
      m_outTemp->write( reinterpret_cast<char *>( &sizes ), sizeof( sizes ) );

      // write the nil entry
      if ( m_switchEntryNil != 0 )
         m_switchEntryNil->write( m_outTemp );
      else {
         int32 dummy = 0xFFFFFFFF;
         m_outTemp->write( reinterpret_cast<char *>( &dummy ), sizeof( dummy ) );
      }

      //write the single entries.
      for ( int32 i = 0; i < 3; i++ )
      {
         Map *pmap;
         switch( i ) {
            case 0: pmap = &m_switchEntriesInt; break;
            case 1: pmap = &m_switchEntriesRng; break;
            case 2: pmap = &m_switchEntriesStr; break;
            //case 3: pmap = &m_switchEntriesObj; break;
         }

         MapIterator iter = pmap->begin();
         while( iter.hasCurrent() )
         {
            Pseudo *first = *(Pseudo**) iter.currentKey();
            Pseudo *second = *(Pseudo**) iter.currentValue();
            first->write( m_outTemp ); // will do the right thing
            second->write( m_outTemp );
            iter.next();
         }
      }

      // for objects, order is relevant.
      ListElement *elem = m_switchObjList.begin();
      while( elem != 0 )
      {
         Pseudo *pseudo = (Pseudo *) elem->data();
         Pseudo **objjmp = (Pseudo **) m_switchEntriesObj.find( pseudo );
         fassert( objjmp != 0 );
         if( objjmp != 0 )
         {
            pseudo->write( m_outTemp ); // will do the right thing
            (*objjmp)->write( m_outTemp );
         }
         elem = elem->next();
      }

      // finally clear the switch
      clearSwitch();
   }
   else {
      raiseError( e_switch_end );
   }

}

bool AsmCompiler::isLocal( Pseudo *op1 ) const
{
   if ( ! m_current || ! m_current->isFunction() )
      return false;

   Symbol *sym = op1->asSymbol();
   if ( sym != 0 && sym->isLocal() )
      return true;
   return false;
}

bool AsmCompiler::isParam( Pseudo *op1 ) const
{
   if ( ! m_current || ! m_current->isFunction() )
      return false;

   // forcing a global symbol?
   if ( op1->asSymbol()->name().getCharAt(0) == '*' )
      return false;

   Symbol *sym = op1->asSymbol();
   if ( sym  != 0 && sym->isParam() )
      return true;
   return false;
}

bool AsmCompiler::isExtern( Pseudo *op1 ) const
{
   Symbol *sym = m_module->symbolTable().findByName( op1->asString() );
   if ( sym !=0 && sym->isUndefined() )
      return true;
   return false;
}


void AsmCompiler::addInstr( unsigned char opcode, Pseudo *op1, Pseudo *op2, Pseudo *op3 )
{
   m_outTemp->write( reinterpret_cast<char *>(&opcode), 1 );

   unsigned char opsp = paramDesc( op1 );
   m_outTemp->write( reinterpret_cast<char *>(&opsp), 1 );
   opsp = paramDesc( op2 );
   m_outTemp->write( reinterpret_cast<char *>(&opsp), 1 );
   opsp = paramDesc( op3 );
   m_outTemp->write( reinterpret_cast<char *>(&opsp), 1 );

   if ( op1 != 0 )
   {
      op1->write( m_outTemp );
      if ( op1->disposeable())
         delete op1;
   }

   if ( op2 != 0 )
   {
      op2->write( m_outTemp );

      if ( op2->disposeable() )
         delete op2;
   }

   if ( op3 != 0 )
   {
      op3->write( m_outTemp );

      if ( op3->disposeable() )
         delete op3;
   }

}



void AsmCompiler::write_switch( Pseudo *op1, Pseudo *op2, Pseudo *oplist )
{
   /*addInstr( P_SWCH, op1, op2 );

   // write also the switch list
   Pseudo::list_t *lst = oplist->asList();
   Pseudo::list_t::iterator iter = lst->begin();
   while( iter != lst->end() ) {
      Pseudo *current = *iter;
      current->write( m_out );
      ++iter;
   }
   delete oplist;
   */
   raiseError(e_syntax, "SWCH opcode not directly implemented in assembly. Use .switch directive" );
}

Symbol *AsmCompiler::findSymbol( const String &name ) const
{
   Symbol *sym = 0;

   // have we got a context to search in?
   if ( m_current != 0 && name.getCharAt( 0 ) != '*' )
   {
      if ( m_current->isFunction() )
         sym = m_current->getFuncDef()->symtab().findByName( name );
      else if( m_current->isClass() )
         sym = m_current->getClassDef()->symtab().findByName( name );
   }

   // havn't we found it in the local table?
   if ( sym == 0 )
      if ( name.getCharAt( 0 ) == '*' )
         sym = m_module->symbolTable().findByName( name.subString( 1 ) );
      else
         sym = m_module->symbolTable().findByName( name );

   return sym;
}


void Pseudo_Deletor( void *pseudo )
{
   Pseudo *val = (Pseudo *) pseudo;

   if ( val->disposeable() )
      delete val;
}

}

/* end of compiler.cpp */
