/*
   FALCON - The Falcon Programming Language.
   FILE: genhasm.cpp

   Generate Falcon Assembly from a Falcon syntactic tree.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>
#include <falcon/genhasm.h>
#include <falcon/syntree.h>
#include <falcon/common.h>
#include <falcon/stream.h>
#include <falcon/genericvector.h>
#include <falcon/traits.h>
#include <falcon/fassert.h>

namespace Falcon
{

GenHAsm::GenHAsm( Stream *out ):
   Generator( out ),
   m_branch_id(1),
   m_loop_id(1),
   m_try_id(1)
{}


void GenHAsm::generatePrologue( const Module *mod )
{
   m_out->writeString( "; -------------------------------------------------------------\n" );
   m_out->writeString( "; Falcon ASM source for module " + mod->name() +  "\n" );
   m_out->writeString( "; -------------------------------------------------------------\n\n" );

   m_out->writeString( "; -------------\n" );
   m_out->writeString( "; String table\n\n" );
   // generates the string table.
   gen_stringTable( mod );
   m_out->writeString( "\n" );

   m_out->writeString( "; -------------\n" );
   m_out->writeString( "; Dependecy table\n\n" );
   // generates the string table.
   gen_depTable( mod );
   m_out->writeString( "\n" );


   // generates the string table.
   m_out->writeString( "; -------------\n" );
   m_out->writeString( "; Symbol table\n\n" );
   gen_symbolTable( mod );
   m_out->writeString( "\n" );

}

void GenHAsm::generate( const SourceTree *st )
{

   // generates the main program
   if ( ! st->statements().empty() )  {
      m_out->writeString( "; -------------\n" );
      m_out->writeString( "; Entry point\n\n" );
      m_out->writeString( ".entry\n" );

      gen_block( &st->statements() );
      m_out->writeString( "\tRET\n" );
   }

   // generate functions
   if ( ! st->classes().empty() ) {

      m_out->writeString( "; -------------------------------------------------------------\n" );
      m_out->writeString( "; Classes\n" );
      m_out->writeString( "; -------------------------------------------------------------\n" );

      const StmtClass *cls = static_cast<const StmtClass*>(st->classes().front());
      while( cls != 0 )
      {
         gen_class( cls );
         cls = static_cast<const StmtClass *>(cls->next());
      }

      m_out->writeString( "\n" );
   }

   // generate functions
   if ( ! st->functions().empty() ) {

      m_out->writeString( "; -------------------------------------------------------------\n" );
      m_out->writeString( "; Routines\n" );
      m_out->writeString( "; -------------------------------------------------------------\n" );

      const StmtFunction *func = static_cast<const StmtFunction*>(st->functions().front());
      while( func != 0 )
      {
         gen_function( func );
         func = static_cast<const StmtFunction *>(func->next());
      }
      m_out->writeString( "\n" );
   }

   m_out->writeString( "\n; -------------------------------------------------------------\n" );
   m_out->writeString( "; End of listing\n" );
   m_out->writeString( "; -------------------------------------------------------------\n" );
}


void GenHAsm::gen_class( const StmtClass *cls )
{
   const Symbol *sym = cls->symbol();
   m_out->writeString( "; -------------------------------------------------------------\n" );
   m_out->writeString( "; Class\n" );
   m_out->writeString( "; -------------------------------------------------------------\n" );
   m_out->writeString( ".classdef " + sym->name() );
   if ( sym->exported() )
      m_out->writeString( " export" );
   m_out->writeString( "\n" );

   const ClassDef *cd = sym->getClassDef();

   // write class symbol parameters.

   MapIterator st_iter = cd->symtab().map().begin();
   int16 count = 0;
   while( st_iter.hasCurrent() && count < cd->symtab().size() )
   {
      // we have no locals.
      const Symbol *param = *( const Symbol **) st_iter.currentValue();
      if( param->itemId() == count )
      {
         m_out->writeString( ".param " + param->name() + "\n" );
         count++;
         st_iter = cd->symtab().map().begin();
         continue;
      }

      st_iter.next();
   }

   // write all the inheritances.
   ListElement *it_elem = cd->inheritance().begin();
   while( it_elem != 0 )
   {
      const InheritDef *id = (const InheritDef *) it_elem->data();
      const Symbol *parent = id->base();
      m_out->writeString( ".inherit $" + parent->name() );
      m_out->writeString( "\n" );
      it_elem = it_elem->next();
   }

   // write all the properties.
   MapIterator pt_iter = cd->properties().begin();
   while( pt_iter.hasCurrent() )
   {
      const VarDef *def = *(const VarDef **) pt_iter.currentValue();
      if ( ! def->isBaseClass() ) {
         if ( def->isReference() )
            m_out->writeString( ".propref " );
         else
            m_out->writeString( ".prop " );
         const String *key = *( const String **) pt_iter.currentKey();
         m_out->writeString( *key + " " );
         gen_propdef( *def );
         m_out->writeString( "\n" );
      }
      pt_iter.next();
   }

   if ( cd->constructor() != 0 )
   {
      m_out->writeString( ".ctor $" + cd->constructor()->name() + "\n" );
   }

   MapIterator state_iter = cd->states().begin();
   while( state_iter.hasCurrent() )
   {
      const StateDef *def = *(const StateDef **) state_iter.currentValue();
      m_out->writeString( ".state " );
      const String *key = *( const String **) state_iter.currentKey();
      m_out->writeString( *key );
      m_out->writeString( "\n" );

      MapIterator s_iter = def->functions().begin();
      while( s_iter.hasCurrent() )
      {
         m_out->writeString( ".stateitem " );
         const String *key = *( const String **) s_iter.currentKey();
         const Symbol *func = *( const Symbol **) s_iter.currentValue();
         m_out->writeString( *key + " $" + func->name() + "\n" );

         s_iter.next();
      }

      state_iter.next();
   }

   m_out->writeString( ".endclass\n" );
}

void GenHAsm::gen_propdef( const VarDef &def )
{
   switch( def.type() )
   {
      case VarDef::t_nil: m_out->writeString( "NIL" ); break;
      case VarDef::t_int:
      {
         String str;
         str.writeNumber( def.asInteger() );
         m_out->writeString( str );
      }
      break;
      case VarDef::t_num:
      {
         String str;
         str.writeNumber( def.asNumeric() );
         m_out->writeString( str );
      }
      break;

      case VarDef::t_string:
      {
         String temp;
         def.asString()->escape( temp );
         m_out->writeString( "\"" + temp + "\"");
      }
      break;
      case VarDef::t_reference:
      case VarDef::t_symbol: m_out->writeString( "$" + def.asSymbol()->name() ); break;

      default:
         break;
   }
}

void GenHAsm::gen_depTable( const Module *mod )
{
   MapIterator iter = mod->dependencies().begin();

   while( iter.hasCurrent() )
   {
      const ModuleDepData *depdata = *(const ModuleDepData **) iter.currentValue();

      // only generate non-private data
      if ( ! depdata->isPrivate() )
         m_out->writeString( ".load " + depdata->moduleName() + "\n" );

      iter.next();
   }
}

void GenHAsm::gen_stringTable( const Module *mod )
{
   uint32 count = 0;
   const String *str = mod->getString( count );
   while( str != 0 ) {
      // for now all cstrings.
      String temp;
      str->escape( temp );
      if ( str->exported() )
         m_out->writeString( ".istring \"" );
      else
         m_out->writeString( ".string \"" );

      m_out->writeString( temp + "\"\n" );
      ++count;
      str = mod->getString( count);
   }
}

void GenHAsm::gen_symbolTable( const Module *mod )
{
   const SymbolTable *symtab = &mod->symbolTable();
   MapIterator iter = symtab->map().begin();
   String temp;

   while( iter.hasCurrent() )
   {
      const Symbol *sym = *(const Symbol **) iter.currentValue();

      switch( sym->type() ) {
         case Symbol::tundef:
            // see if it's imported
            if ( sym->imported() )
            {
               // see if we have a namespace into which import it
               uint32 dotpos = sym->name().rfind( "." );
               if( dotpos != String::npos )
               {
                  String modname = sym->name().subString( 0, dotpos );
                  String symname = sym->name().subString( dotpos + 1 );

                  temp =  ".import " + symname + " ";
                  temp.writeNumber( (int64) sym->declaredAt() );

                  ModuleDepData *depdata = mod->dependencies().findModule( modname );
                  // We have created the module, the entry must be there.
                  fassert( depdata != 0 );
                  if ( depdata->isFile() )
                  {
                     if( depdata->moduleName() == modname )
                        temp += " \"" + modname+"\"";
                     else
                        temp += " \"" + depdata->moduleName() +"\" "+ modname;
                  }
                  else
                  {
                     if( depdata->moduleName() == modname )
                        temp += " " + modname;
                     else
                        temp += " " + depdata->moduleName() +" "+ modname;
                  }
               }
               else {
                  temp =  ".import " + sym->name() + " ";
                  temp.writeNumber( (int64) sym->declaredAt() );
               }
            }
            else {
               temp =  ".extern " + sym->name() + " ";
               temp.writeNumber( (int64) sym->declaredAt() );
            }
            m_out->writeString( temp );
         break;

         case Symbol::tglobal:
            temp =  ".global " + sym->name() + " ";
            temp.writeNumber( (int64) sym->declaredAt() );
            m_out->writeString( temp );
         break;

         case Symbol::tvar:
            m_out->writeString( ".var " + sym->name() + " " );
            gen_propdef( *sym->getVarDef() );
            temp = " ";
            temp.writeNumber( (int64) sym->declaredAt() );
            m_out->writeString( temp );
         break;

         case Symbol::tconst:
            m_out->writeString( ".const " + sym->name() + " " );
            gen_propdef( *sym->getVarDef() );
            temp = " ";
            temp.writeNumber( (int64) sym->declaredAt() );
            m_out->writeString( temp );
         break;

         case Symbol::tfunc:
            temp =  ".func " + sym->name() + " ";
            temp.writeNumber( (int64) sym->declaredAt() );
            m_out->writeString( temp );
         break;

         case Symbol::tclass:
            temp =  ".class " + sym->name() + " ";
            temp.writeNumber( (int64) sym->declaredAt() );
            m_out->writeString( temp );
         break;
         case Symbol::tinst:
            m_out->writeString( ".instance $" + sym->getInstance()->name() + " " +
               sym->name() );
            temp = " ";
            temp.writeNumber( (int64) sym->declaredAt() );
            m_out->writeString( temp );
         break;

         case Symbol::timportalias:
            {
               ImportAlias* ia = sym->getImportAlias();
               temp =  ".alias " + ia->name() + " ";
               temp.writeNumber( (int64) sym->declaredAt() );
               temp.append( " " );
               if ( ia->isOrigFileName() ) temp.append( "\"" );
               temp.append( ia->origModule() );
               if ( ia->isOrigFileName() ) temp.append( "\"" );
               temp.append( " " );
               temp.append( sym->name() );
            }
            m_out->writeString( temp );
         break;

         default:
            break;

      }

      if ( sym->exported() && ! sym->isUndefined() )
         m_out->writeString( " export" );
      m_out->writeString( "\n" );

      iter.next();
   }
}

void GenHAsm::gen_function( const StmtFunction *func )
{
   m_functions.pushBack( func->symbol() );
   const StmtClass *ctorFor = func->constructorFor();
   const char *ret_mode = ctorFor != 0 ? "\tRETV\tS1\n" : "\tRET\n";

   m_out->writeString( "; ---------------------------------------------\n" );
   m_out->writeString( "; Function " + func->symbol()->name() + "\n" );
   m_out->writeString( "; ---------------------------------------------\n\n" );
   m_out->writeString( ".funcdef " + func->symbol()->name() );
   if ( func->symbol()->exported() )
      m_out->writeString( " export" );
   m_out->writeString( "\n" );

   // generate the local symbol table.
   const FuncDef *fd = func->symbol()->getFuncDef();
   GenericVector params( &traits::t_voidp() );
   GenericVector locals( &traits::t_voidp() );

   params.resize( fd->symtab().size() );
   locals.resize( fd->symtab().size() );


   MapIterator iter = fd->symtab().map().begin();
   while( iter.hasCurrent() )
   {
      Symbol *sym = *(Symbol **) iter.currentValue();
      switch( sym->type() ) {
         case Symbol::tparam:
            // paramters must be outputted with their original order.
            params.set( sym, sym->itemId() );
         break;
         case Symbol::tlocal:
            locals.set( sym, sym->itemId() );

         break;

         default:
            break;
      }

      iter.next();
   }

   for ( uint32 parId = 0; parId < params.size(); parId++ )
   {
      Symbol *sym = *(Symbol **) params.at( parId );
      if (sym != 0 )
         m_out->writeString( ".param " + sym->name() + "\n" );
   }

   for ( uint32 locId = 0; locId < locals.size(); locId++ )
   {
      Symbol *sym = *(Symbol **) locals.at( locId );
      if (sym != 0 )
         m_out->writeString( ".local " + sym->name() + "\n" );
   }

   // generates INIT for constructors
   if ( ctorFor != 0 )
   {
      ListElement *it_iter = ctorFor->initExpressions().begin();
      while( it_iter != 0 )
      {
         const Value *value = (const Value *) it_iter->data();
         gen_value( value );
         it_iter = it_iter->next();
      }
   }

   const char *end_type = ret_mode;

   if ( func->statements().empty() ) {
      if ( func->staticBlock().empty() ) {
         m_out->writeString( end_type );
      }
      else {
         m_out->writeString( "\tONCE\t _" + func->name() + "_once_end, $" + func->name() + "\n" );
         gen_block( &func->staticBlock() );
         m_out->writeString( "_" + func->name() + "_once_end:" + "\n" );
         if( func->staticBlock().back()->type() != Statement::t_return ) {
            m_out->writeString( end_type );
         }
      }
   }
   else {
      if ( ! func->staticBlock().empty() ) {
         m_out->writeString( "\tONCE\t _" + func->name() + "_once_end, $" + func->name() + "\n" );
         gen_block( &func->staticBlock() );  // ok also if empty
         m_out->writeString( "_" + func->name() + "_once_end:" + "\n" );
      }

      gen_block( &func->statements() );
      if( func->statements().back()->type() != Statement::t_return ) {
         m_out->writeString( end_type );
      }
   }

   m_out->writeString( ".endfunc\n\n" );
   m_functions.popBack();
}

void GenHAsm::gen_block( const StatementList *slist )
{
   const Statement *stmt = slist->front();
   while( stmt != 0 ) {
      gen_statement( stmt );
      stmt = static_cast<const Statement *>(stmt->next());
   }
}

void GenHAsm::gen_statement( const Statement *stmt )
{
   static uint32 last_line=0;
   if ( stmt->line() > 0 && last_line != stmt->line() ) {
      last_line = stmt->line();
      String linestr;
      linestr.writeNumber( (int64) last_line );
      m_out->writeString( ".line " + linestr + "\n" );
   }

   switch( stmt->type() )
   {
      case Statement::t_none:
         // this must be ignored.
      break;

      case Statement::t_break:
      case Statement::t_continue:
      {
         fassert( ! m_loops.empty() );
         LoopInfo* loop = (LoopInfo*)  m_loops.back();
         int target_pos = loop->m_id;


         // there is a TRY statement above this one. If the TRY level is "inner"
         // ( try is inside the loop we are going to break ) we have to also pop
         // the TRY value while restarting the loop. If the TRY is outside the loop,
         // then everything stays the same.
         ListElement *trypos = m_trys.end();
         int steps = 0;
         while( trypos != 0 ) {
            if ( ((int) trypos->iData()) >= target_pos )
               steps ++;
            trypos = trypos->prev();
         }
         String stepsStr;
         stepsStr.writeNumber( (int64) steps );

         if ( steps > 0 )
            m_out->writeString( "\tPTRY\t" + stepsStr + "\n" );

         String loopStr;
         loopStr.writeNumber( (int64) target_pos );
         if ( stmt->type() == Statement::t_continue )
         {
            const StmtContinue *cont = static_cast<const StmtContinue *>( stmt );
            // is a continue dropping? -- we must generate a TRDN opcode.
            if( cont->dropping() )
            {
               fassert( loop->m_loop->type() == Statement::t_forin );

               // when generating the last element, TRDN is simpler.
               if( loop->m_isForLast )
               {
                  m_out->writeString( "\tTRDN\t_loop_end_" + loopStr + ", 0, 0\n" );
               }
               else
               {
                  StmtForin* fin = (StmtForin* ) loop->m_loop;
                  String svars;
                  svars.writeNumber((int64) fin->dest()->size() );
                  m_out->writeString( "\tTRDN\t_loop_begin_" + loopStr + ", " +
                                      "_loop_end_" + loopStr +
                                      ", "+ svars +"\n" );
                  ListElement *it_t = fin->dest()->begin();

                  // first generates an array of references
                  while( it_t != 0 ) {
                     // again, is the compiler that must make sure of this...
                     const Value *val = (const Value *) it_t->data();
                     fassert( val->isSimple() );

                     m_out->writeString( "\tNOP \t" );
                     gen_operand( val );
                     m_out->writeString( "\n" );

                     it_t = it_t->next();
                  }
               }
            }
            else {
               // when generating the last element, continue == break.
               if( loop->m_isForLast )
                  m_out->writeString( "\tJMP \t_loop_end_" + loopStr + "\n" );
               else
                  m_out->writeString( "\tJMP \t_loop_next_" + loopStr + "\n" );
            }
         }
         else
            m_out->writeString( "\tJMP \t_loop_end_" + loopStr + "\n" );
      }
      break;

      case Statement::t_launch:
      {
         const StmtLaunch *launch = static_cast<const StmtLaunch *>( stmt );
         const Value *call = launch->value();
         fassert( call->isExpr() );
         const Expression *expr = call->asExpr();
         fassert( expr->type() == Expression::t_funcall );
         gen_funcall( expr, true );
      }
      break;

      case Statement::t_autoexp:
      {
         const StmtExpression *val = static_cast<const StmtExpression *>( stmt );
         if ( ! val->value()->isSimple() ) {
            gen_complex_value( val->value() );
         }
      }
      break;

      case Statement::t_return:
      {
         const StmtReturn *ret = static_cast<const StmtReturn *>( stmt );

         if ( ret->value() == 0 ) {
            m_out->writeString( "\tRET \t" );
         }
         else if ( ret->value()->isSimple() ) {
            m_out->writeString( "\tRETV\t" );
            gen_operand( ret->value() );
         }
         else {
            gen_complex_value( ret->value() );
            m_out->writeString( "\tRETA\t" );
         }
         m_out->writeString( "\n" );
      }
      break;

      case Statement::t_raise:
      {
         const StmtRaise *op = static_cast< const StmtRaise *>( stmt );

         if ( op->value()->isSimple() )
         {
            m_out->writeString( "\tRIS \t" );
            gen_operand( op->value() );
         }
         else {
            gen_complex_value( op->value() );
            m_out->writeString( "\tRIS \tA" );
         }
         m_out->writeString( "\n" );
      }
      break;

      case Statement::t_fordot:
      {
         const StmtFordot *op = static_cast< const StmtFordot *>( stmt );

         if ( op->value()->isSimple() )
         {
            m_out->writeString( "\tTRAC\t" );
            gen_operand( op->value() );
         }
         else {
            gen_complex_value( op->value() );
            m_out->writeString( "\tTRAC\tA" );
         }
         m_out->writeString( "\n" );
      }
      break;


      case Statement::t_global:
      {
         // ignore the statement.
      }
      break;

      case Statement::t_self_print:
      {
         const StmtSelfPrint *sp = static_cast<const StmtSelfPrint *>( stmt );
         const ArrayDecl *attribs = sp->toPrint();
         ListElement *iter = attribs->begin();

         while( iter != 0 )
         {
            const Value *val = (const Value *) iter->data();
            if ( val->isSimple() )
            {
               m_out->writeString( "\tWRT \t" );
               gen_operand( val );
            }
            else {
               gen_complex_value( val );
               m_out->writeString( "\tWRT \tA" );
            }
            m_out->writeString( "\n" );

            iter = iter->next();
         }
      }
      break;

      case Statement::t_unref:
      {
         const StmtUnref *ass = static_cast<const StmtUnref *>(stmt);
         if( ass->symbol()->isSimple() ) {
            m_out->writeString( "\tLDRF\t" );
            gen_operand( ass->symbol() );
            m_out->writeString( ", 0\n" );
         }
         else {
            gen_complex_value( ass->symbol() );
            m_out->writeString( "\tLDRF\tA, 0\n" );
         }
      }
      break;

      case Statement::t_if:
      {
         const StmtIf *elem = static_cast<const StmtIf *>( stmt );

         if ( elem->children().empty() && elem->elifChildren().empty() &&
               elem->elseChildren().empty() ) {
            if ( ! elem->condition()->isSimple() )
               gen_complex_value( elem->condition() );  // generate & discard value
            break; // nothing more needed
         }

         int branch = m_branch_id++;
         String branchStr;
         branchStr.writeNumber( (int64) branch );
         m_branches.pushBack( (void *) branch );
         gen_condition( elem->condition() );
         if( ! elem->children().empty() ) {
            gen_block( &elem->children() );
            // do we need to jump away?
         }
         if( ! (elem->elifChildren().empty() && elem->elseChildren().empty() ) ) {
            m_out->writeString( "\tJMP \t_branch_end_" + branchStr + "\n" );
         }
         m_out->writeString( "_branch_fail_" + branchStr + ":\n" );

         if ( ! elem->elifChildren().empty() )
         {
            const StmtElif *selif = static_cast<const StmtElif *>(elem->elifChildren().front());
            int elifcount = 1;
            while( selif != 0 ) {
               gen_condition( selif->condition(), elifcount );
               if( ! selif->children().empty() ) {
                  gen_block( &selif->children() );
                  // do we need to jump away?
                  if( selif->next() != 0 || ! elem->elseChildren().empty() ) {
                     m_out->writeString( "\tJMP \t_branch_end_" + branchStr + "\n" );
                  }
                  String elifStr;
                  elifStr.writeNumber( (int64) elifcount );
                  m_out->writeString( "_branch_" + branchStr + "_elif_");
                  m_out->writeString( elifStr + ":\n" );
               }
               elifcount++;
               selif = static_cast<const StmtElif *>(selif->next());
            }
         }

         if ( ! elem->elseChildren().empty() ) {
            gen_block( &elem->elseChildren() );
         }

         // have we any block to jump across?
         m_out->writeString( "_branch_end_" + branchStr + ":\n" );

         m_branches.popBack();
      }
      break;

      case Statement::t_switch:
      case Statement::t_select:
      {
         const StmtSwitch *elem = static_cast<const StmtSwitch *>( stmt );
         // just check for the item to be not empty
         if ( elem->intCases().empty() && elem->rngCases().empty() &&
              elem->strCases().empty() && elem->objCases().empty() &&
              elem->defaultBlock().empty() &&
              elem->nilBlock() == -1 )
         {
            if ( ! elem->switchItem()->isSimple() )
               gen_complex_value( elem->switchItem() );
            break;
         }

         const char *oper = stmt->type() == Statement::t_switch ? ".switch " : ".select ";
         if ( elem->switchItem()->isSimple() )
         {
            m_out->writeString( oper );
            gen_operand( elem->switchItem() );
         }
         else {
            gen_complex_value( elem->switchItem() );
            m_out->writeString( oper );
            m_out->writeString( "A" );
         }
         int branch = m_branch_id++;
         String branchStr;
         branchStr.writeNumber( (int64) branch );


         if ( elem->defaultBlock().empty() )
            m_out->writeString( ", _switch_" + branchStr + "_end" );
         else
            m_out->writeString( ", _switch_" + branchStr + "_default" );

         m_out->writeString( "\n" );
         if ( elem->nilBlock() != -1 ) {
            String caseStr;
            caseStr.writeNumber( (int64) elem->nilBlock() );
            m_out->writeString( ".case NIL, _switch_" + branchStr + "_case_");
            m_out->writeString( caseStr + "\n" );
         }
         dump_cases( branch, elem->intCases().begin() );
         dump_cases( branch, elem->rngCases().begin() );
         dump_cases( branch, elem->strCases().begin() );

         // we must put the objects in the same order they were declared.
         ListElement *iter = elem->objList().begin();
         while( iter != 0 )
         {
            Value *val = (Value *) iter->data();
            MapIterator case_iter;
            if( elem->objCases().find( val, case_iter ) )
            {
               String caseStr;
               int64 cn = (int64)  *(uint32*) case_iter.currentValue();
               caseStr.writeNumber( cn );
               m_out->writeString( ".case " );
               gen_operand( val );
               m_out->writeString( ", _switch_" + branchStr + "_case_" );
               m_out->writeString( caseStr + "\n" );
            }
            iter = iter->next();
         }

         m_out->writeString( ".endswitch\n\n" );

         // generate the code blocks
         stmt = elem->blocks().front();
         int id = 0;
         while( stmt != 0 ) {
            const StmtCaseBlock *elem_case = static_cast<const StmtCaseBlock *>( stmt );
            // skip default block.
            String idStr;
            idStr.writeNumber( (int64) id );
            m_out->writeString( "_switch_" + branchStr + "_case_" );
            m_out->writeString( idStr + ":\n" );
            gen_block( &elem_case->children() );
            // jump away, but only if needed.
            if ( elem_case->next() != 0 || ! elem->defaultBlock().empty() )
               m_out->writeString( "\tJMP \t_switch_" + branchStr + "_end\n" );
            id++;
            stmt = static_cast<const Statement *>(elem_case->next());
         }
         if ( ! elem->defaultBlock().empty() ) {
            m_out->writeString( "_switch_" + branchStr + "_default:\n" );
            gen_block( &elem->defaultBlock() );
         }
         m_out->writeString( "_switch_" + branchStr + "_end:\n" );
      }
      break;

      case Statement::t_while:
      {
         const StmtWhile *elem = static_cast<const StmtWhile *>( stmt );

         int branch = m_loop_id++;
         LoopInfo loop( branch, elem );
         m_loops.pushBack( (void *) &loop );
         String branchStr;
         branchStr.writeNumber( (int64) branch );
         m_out->writeString( "_loop_next_" + branchStr + ":\n" );
         m_out->writeString( "_loop_begin_" + branchStr + ":\n" );

         // Generate the condition only if present and not always true.
         if ( elem->condition() != 0 && ! elem->condition()->isTrue() ) {
            if ( elem->condition()->isSimple() ) {
               m_out->writeString( "\tIFF \t_loop_end_" + branchStr + ", " );
               gen_operand( elem->condition() );
            }
            else {
               gen_complex_value( elem->condition() );
               m_out->writeString( "\tIFF \t_loop_end_" + branchStr + ", A" );
            }
            m_out->writeString( "\n" );
         }

         gen_block( &elem->children() );

         m_out->writeString( "\tJMP \t_loop_begin_" + branchStr + "\n" );
         m_out->writeString( "_loop_end_" + branchStr + ":\n" );
         m_loops.popBack();
      }
      break;

      case Statement::t_loop:
      {
         const StmtLoop *elem = static_cast<const StmtLoop* >( stmt );

         int branch = m_loop_id++;
         LoopInfo loop( branch, elem );
         m_loops.pushBack( (void *) &loop );
         String branchStr;
         branchStr.writeNumber( (int64) branch );
         m_out->writeString( "_loop_begin_" + branchStr + ":\n" );

         if ( elem->condition() == 0 )
            m_out->writeString( "_loop_next_" + branchStr + ":\n" );

         gen_block( &elem->children() );

         if ( elem->condition() == 0 )
         {
            // endless loop
            m_out->writeString( "\tJMP \t_loop_begin_" + branchStr + "\n" );
         }
         else {
            m_out->writeString( "_loop_next_" + branchStr + ":\n" );

            if ( ! elem->condition()->isTrue() )
            {
               if ( elem->condition()->isSimple() ) {
                  m_out->writeString( "\tIFF \t_loop_begin_" + branchStr + ", " );
                  gen_operand( elem->condition() );
               }
               else {
                  gen_complex_value( elem->condition() );
                  m_out->writeString( "\tIFF \t_loop_begin_" + branchStr + ", A" );
               }
               m_out->writeString( "\n" );
            }
         }
         // if it's true, terminate immediately

         m_out->writeString( "_loop_end_" + branchStr + ":\n" );
         m_loops.popBack();
      }
      break;

      case Statement::t_propdef:
      {
         const StmtVarDef *pdef = static_cast<const StmtVarDef *>( stmt );
         if ( pdef->value()->isSimple() ) {
            m_out->writeString( "\tSTP \tS1, \"" + *pdef->name() + "\", " );
            gen_operand( pdef->value() );
         }
         else {
            gen_value( pdef->value() );
            m_out->writeString( "\tSTP \tS1, \"" + *pdef->name() + "\", A" );
         }
         m_out->writeString( "\n" );
      }
      break;

      case Statement::t_forin:
      {
         const StmtForin *loop = static_cast<const StmtForin *>( stmt );

         int loopId = m_loop_id++;

         LoopInfo li( loopId, loop );
         m_loops.pushBack( (void *) &li );

         String loopStr;
         loopStr.writeNumber( (int64) loopId );

         String snv;
         snv.writeNumber( (int64) loop->dest()->size() );

         if ( loop->source()->isSimple() ) {
            m_out->writeString( "\tTRAV\t_p_loop_end_" + loopStr + ", " );
            m_out->writeString( snv + ", " );
            gen_operand( loop->source() );
            m_out->writeString( "\n" );
         }
         else {
            gen_value( loop->source() );
            m_out->writeString( "\tTRAV\t_p_loop_end_" + loopStr + ", " );
            m_out->writeString( snv + ", A\n" );
         }

         ListElement* it_t = loop->dest()->begin();

         // first generates an array of references
         while( it_t != 0 ) {
            // again, is the compiler that must make sure of this...
            const Value *val = (const Value *) it_t->data();
            fassert( val->isSimple() );

            m_out->writeString( "\tNOP \t" );
            gen_operand( val );
            m_out->writeString( "\n" );

            it_t = it_t->next();
         }

         // have we got a "first" block?
         if ( ! loop->firstBlock().empty() ) {
            gen_block( &loop->firstBlock() );
         }

         // begin of the main loop;
         m_out->writeString( "_loop_begin_" + loopStr + ":\n" );
         if( ! loop->children().empty() ) {
            gen_block( &loop->children() );
         }

         // generate the middle block
         if( ! loop->middleBlock().empty() ) {
            // skip it for the last element
            m_out->writeString( "\tTRAL\t_loop_tral_" + loopStr + "\n" );
            gen_block( &loop->middleBlock() );
         }

         m_out->writeString( "_loop_next_" + loopStr + ":\n" );
         m_out->writeString( "\tTRAN\t_loop_begin_" + loopStr + ", "+ snv +",\n");
         it_t = loop->dest()->begin();

         // first generates an array of references
         while( it_t != 0 ) {
           // again, is the compiler that must make sure of this...
           const Value *val = (const Value *) it_t->data();
           fassert( val->isSimple() );

           m_out->writeString( "\tNOP \t" );
           gen_operand( val );
           m_out->writeString( "\n" );

           it_t = it_t->next();
         }

         // generate the last block
         m_out->writeString( "_loop_tral_" + loopStr + ":\n" );
         if( ! loop->lastBlock().empty() ) {
            // and the last time...
            li.m_isForLast = true;
            gen_block( &loop->lastBlock() );
         }

         // create break landing code:
         m_out->writeString( "_loop_end_" + loopStr + ":\n" );
         m_out->writeString( "\tIPOP\t1\n" );
         // internal loop out used by TRAV, TRAN and TRAL
         m_out->writeString( "_p_loop_end_" + loopStr + ":\n" );

         m_loops.popBack();
      }
      break;

      case Statement::t_try:
      {
         const StmtTry *op = static_cast< const StmtTry *>( stmt );
         // if the try block is empty we have nothing to do
         if( op->children().empty() )
            break;
         // push the LOOP id that may cause a TRY breaking.
         if ( m_loops.empty() )
            m_trys.pushBack( (void *) -1 );
         else
            m_trys.pushBack( ((LoopInfo*) m_loops.back())->m_id );

         int branch = m_try_id++;
         String branchStr;
         branchStr.writeNumber( (int64) branch );
         // as TRY does not have a condition, and everyone is on
         // its own, we don't have to push it as a branch.

         m_out->writeString( "\tTRY \t_branch_try_" + branchStr + "\n" );

         // MUST maintain the current branch level to allow inner breaks.
         gen_block( &op->children() );

         // When the catcher is generated, the TRY cannot be broken anymore
         // by loop controls, as the catcher pops the TRY context from the VM
         m_trys.popBack();

         // now generate the catch blocks
         // if we have no default nor specific blocs, we're done
         if ( op->handlers().empty() && ! op->defaultGiven() )
         {
            m_out->writeString( "\tPTRY \t1\n" );
            m_out->writeString( "_branch_try_" + branchStr + ":\n" );
            break;
         }

         // if we have only the default block, we don't have a to generate a select.
         if ( op->handlers().empty() )
         {
            m_out->writeString( "\tJTRY \t_branch_try_end_" + branchStr + "\n" );
            m_out->writeString( "_branch_try_" + branchStr + ":\n" );

            if ( op->defaultHandler()->intoValue() != 0 )
               gen_load_from_reg( op->defaultHandler()->intoValue(), "B" );
            gen_block( &op->defaultHandler()->children() );

            m_out->writeString( "_branch_try_end_" + branchStr + ":\n" );
            break;
         }

         // great, we have to create a select B statement.
         m_out->writeString( "\tJTRY \t_branch_try_end_" + branchStr + "\n" );
         m_out->writeString( "_branch_try_" + branchStr + ":\n" );

         m_out->writeString( ".select B" );
         int branch_select = m_branch_id++;
         String branch_selectStr;
         branch_selectStr.writeNumber( (int64) branch_select );

         if ( op->defaultHandler() == 0 )
            m_out->writeString( ", _switch_" + branch_selectStr + "_end" );
         else
            m_out->writeString( ", _switch_" + branch_selectStr + "_default" );

         m_out->writeString( "\n" );
         dump_cases( branch_select, op->intCases().begin() );

         // we must put the objects in the same order they were declared.
         ListElement *iter = op->objList().begin();
         while( iter != 0 )
         {
            Value *val = (Value *) iter->data();
            MapIterator case_iter;
            if( op->objCases().find( val, case_iter ) )
            {
               String caseStr;
               int64 cn = (int64)  *(uint32*) case_iter.currentValue();
               caseStr.writeNumber( cn );
               m_out->writeString( ".case " );
               gen_operand( val );
               m_out->writeString( ", _switch_" + branch_selectStr + "_case_" );
               m_out->writeString( caseStr + "\n" );
            }
            iter = iter->next();
         }

         m_out->writeString( ".endswitch\n\n" );

         // generate the code blocks
         stmt = op->handlers().front();
         int id = 0;
         while( stmt != 0 ) {
            const StmtCatchBlock *elem_catch = static_cast<const StmtCatchBlock *>( stmt );
            // skip default block.
            String idStr;
            idStr.writeNumber( (int64) id );
            m_out->writeString( "_switch_" + branch_selectStr + "_case_" );
            m_out->writeString( idStr + ":\n" );
            if ( elem_catch->intoValue() != 0 )
               gen_load_from_reg( elem_catch->intoValue(), "B" );

            gen_block( &elem_catch->children() );
            // jump away, but only if needed.
            if ( elem_catch->next() != 0 || op->defaultHandler() != 0 )
               m_out->writeString( "\tJMP \t_switch_" + branch_selectStr + "_end\n" );
            id++;
            stmt = static_cast<const Statement *>(elem_catch->next());
         }

         if ( op->defaultHandler() != 0 ) {
            m_out->writeString( "_switch_" + branch_selectStr + "_default:\n" );
            if ( op->defaultHandler() ->intoValue() != 0 )
               gen_load_from_reg( op->defaultHandler()->intoValue(), "B" );
            gen_block( &op->defaultHandler()->children() );
         }
         m_out->writeString( "_switch_" + branch_selectStr + "_end:\n" );
         m_out->writeString( "_branch_try_end_" + branchStr + ":\n" );
      }
      break;

      default:
         break;
   }
}


void GenHAsm::dump_cases( int branch, const MapIterator &begin1 )
{
   MapIterator begin = begin1;
   while( begin.hasCurrent() )
   {
      Value *val = *(Value **) begin.currentKey();
      uint32 id = *(uint32 *) begin.currentValue();

      m_out->writeString( ".case " );

      if ( val->isRange() )
      {
         String start, end;
         start.writeNumber( val->asRange()->rangeStart()->asInteger() );
         end.writeNumber( val->asRange()->rangeEnd()->asInteger() );
         m_out->writeString( start + ":" + end );
      }
      else
         gen_operand( val );

      String branchStr, idStr;
      branchStr.writeNumber( (int64) branch );
      idStr.writeNumber( (int64) id );
      m_out->writeString( ", _switch_" + branchStr + "_case_" );
      m_out->writeString( idStr + "\n" );

      begin.next();
   }
}


void GenHAsm::gen_inc_prefix( const Value *val )
{
   if ( val->isSimple() ) {
      m_out->writeString( "\tINC \t" );
      gen_operand( val );
      m_out->writeString( "\n" );
   }
   else {
      t_valType xValue = l_value;
      gen_complex_value( val, xValue );
      m_out->writeString( "\tINC \tA\n" );
      if ( xValue == p_value )
         m_out->writeString( "\tSTP \tL1, L2, A\n" );
      else if ( xValue == v_value )
         m_out->writeString( "\tSTV \tL1, L2, A\n" );
   }
}

void GenHAsm::gen_inc_postfix( const Value *val )
{
   if ( val->isSimple() ) {
      m_out->writeString( "\tINCP\t" );
      gen_operand( val );
      m_out->writeString( "\n" );
   }
   else {
      t_valType xValue = l_value;
      gen_complex_value( val, xValue );
      m_out->writeString( "\tINCP\tA\n" );

      if ( xValue == p_value )
         m_out->writeString( "\tSTP \tL1, L2, B\n" );
      else if ( xValue == v_value )
         m_out->writeString( "\tSTV \tL1, L2, B\n" );
   }
}

void GenHAsm::gen_dec_prefix( const Value *val )
{
   if ( val->isSimple() ) {
      m_out->writeString( "\tDEC \t" );
      gen_operand( val );
      m_out->writeString( "\n" );
   }
   else {
      t_valType xValue = l_value;
      gen_complex_value( val, xValue );

      m_out->writeString( "\tDEC \tA\n" );

      if ( xValue == p_value )
         m_out->writeString( "\tSTP \tL1, L2, A\n" );
      else if ( xValue == v_value )
         m_out->writeString( "\tSTV \tL1, L2, A\n" );
   }
}

void GenHAsm::gen_dec_postfix( const Value *val )
{
   if ( val->isSimple() ) {
      m_out->writeString( "\tDECP\t" );
      gen_operand( val );
      m_out->writeString( "\n" );
   }
   else {
      t_valType xValue = l_value;
      gen_complex_value( val, xValue );

      m_out->writeString( "\tDECP\tA\n" );

      if ( xValue == p_value )
         m_out->writeString( "\tSTP \tL1, L2, B\n" );
      else if ( xValue == v_value )
         m_out->writeString( "\tSTV \tL1, L2, B\n" );
   }
}

void GenHAsm::gen_autoassign( const char *op, const Value *target, const Value *source )
{
   String opstr = op;
   if( target->isSimple() && source->isSimple() ) {
      m_out->writeString( "\t" + opstr + "\t" );
      gen_operand( target );
      m_out->writeString( ", " );
      gen_operand( source );
      m_out->writeString( "\n" );
   }
   else if ( target->isSimple() )
   {
      gen_complex_value( source );
      m_out->writeString( "\t" + opstr + "\t" );
      gen_operand( target );
      m_out->writeString( ", A\n" );
   }
   else if ( source->isSimple() )
   {
      t_valType xValue = l_value;
      gen_complex_value( target, xValue );

      m_out->writeString( "\t" + opstr + "\tA, " );
      gen_operand( source );
      m_out->writeString( "\n" );

      if ( xValue == p_value )
         m_out->writeString( "\tSTP \tL1, L2, A\n" );
      else if ( xValue == v_value )
         m_out->writeString( "\tSTV \tL1, L2, A\n" );
   }
   else {
      gen_complex_value( source );
      m_out->writeString( "\tPUSH\tA\n" );

      t_valType xValue = l_value;
      gen_complex_value( target, xValue );

      m_out->writeString( "\tPOP \tB\n" );
      m_out->writeString( "\t" + opstr + "\tA, B\n" );

      if ( xValue == p_value )
         m_out->writeString( "\tSTP \tL1, L2, A\n" );
      else if ( xValue == v_value )
         m_out->writeString( "\tSTV \tL1, L2, A\n" );
   }
}

void GenHAsm::gen_condition( const Value *stmt, int mode )
{
   if ( !stmt->isSimple() )
   {
      gen_complex_value( stmt );
   }

   String branchStr;
   branchStr.writeNumber( (int64) m_branches.back() );


   m_out->writeString( "\tIFF \t" );
   if ( mode > 0 ) {
      String modeStr;
      modeStr.writeNumber( (int64) mode );
      m_out->writeString( "_branch_" + branchStr + "_elif_" + modeStr );
   }
   else
      m_out->writeString( "_branch_fail_" + branchStr );

   if ( stmt->isSimple() ) {
      m_out->writeString( ", " );
      gen_operand( stmt );
   }
   else {
      m_out->writeString( ", A" );
   }
   m_out->writeString( "\n" );
}

void GenHAsm::gen_value( const Value *stmt, const char *prefix, const char *cpl_post )
{
   if ( prefix == 0 )
      prefix = "\tLD  \tA, ";

   if ( stmt->isSimple() ) {
      m_out->writeString( prefix );
      gen_operand( stmt );
   }
   else {
      gen_complex_value( stmt );

      if ( cpl_post != 0 )
         m_out->writeString( cpl_post );
         m_out->writeString( "\n" );
   }
}


void GenHAsm::gen_complex_value( const Value *stmt, t_valType &xValue )
{
   switch( stmt->type() )
   {
      // catch also reference taking in case it's not filtered before
      // this happens when we have not a specific opcode to handle references
      case Value::t_byref:
         if ( stmt->asReference()->isSymbol() )
            m_out->writeString( "\tLDRF\tA, $" + stmt->asReference()->asSymbol()->name() + "\n" );
         else {
            gen_value( stmt->asReference() );
            // won't do a lot, but we need it
            m_out->writeString( "\tLDRF\tA, A\n" );
         }
      break;

      case Value::t_array_decl:
         gen_array_decl( stmt->asArray() );
      break;

      case Value::t_dict_decl:
         gen_dict_decl( stmt->asDict() );
      break;

      case Value::t_expression:
         gen_expression( stmt->asExpr(), xValue );
      break;

      case Value::t_range_decl:
         gen_range_decl( stmt->asRange() );
      break;

      default:
         m_out->writeString( "; can't generate this\n" );
   }
}


void GenHAsm::gen_push( const Value *val )
{
   if( val->isSimple() ) {
      m_out->writeString( "\tPUSH\t" );
      gen_operand( val );
   }
   else if ( val->isReference() ) {
      if( val->asReference()->isSymbol() )
         m_out->writeString( "\tPSHR\t$" + val->asReference()->asSymbol()->name() );
      else {
         gen_value( val->asReference() );
         m_out->writeString( "\tPSHR\tA" );
      }
   }
   else {
      gen_complex_value( val );
      m_out->writeString( "\tPUSH\tA" );
   }
   m_out->writeString( "\n" );
}

void GenHAsm::gen_operand( const Value *stmt )
{
   switch( stmt->type() )
   {
      case Value::t_nil:
         m_out->writeString( "NIL" );
         break;

      case Value::t_unbound:
         m_out->writeString( "UNB" );
         break;

      case Value::t_symbol:
      {
         const Symbol *sym = stmt->asSymbol();
         m_out->writeString( "$" );
         if ( ! m_functions.empty() && sym->isGlobal() )
            m_out->writeString( "*" );
         m_out->writeString(  sym->name() );
      }
      break;

      case Value::t_lbind:
         m_out->writeString( "&"+ *stmt->asLBind() );
      break;

      case Value::t_imm_bool:
         m_out->writeString( stmt->asBool() ? "T" : "F" );
      break;

      case Value::t_imm_integer:
      {
         String intStr;
         intStr.writeNumber( stmt->asInteger() );
         m_out->writeString( intStr );
      }
      break;

      case Value::t_imm_num:
      {
         String numStr;
         numStr.writeNumber( stmt->asNumeric(), "%.12g" );
         m_out->writeString( numStr );
      }
      break;

      case Value::t_imm_string:
      {
         String temp;
         stmt->asString()->escape( temp );
         m_out->writeString( "\"" + temp + "\"" );
      }
      break;

      case Value::t_self:
         m_out->writeString( "S1" );
      break;

      default:
         m_out->writeString( "???" );
   }
}

void GenHAsm::gen_expression( const Expression *exp, t_valType &xValue )
{

   String opname;
   int mode = 0; // 1 = unary, 2 = binary

   // first, deterime the operator name and operation type
   switch( exp->type() )
   {
      case Expression::t_none:
         return;

      // optimized away operations
      case Expression::t_optimized:
         gen_value( exp->first() );
         // nothing else needed, going out
      return;

      // logical connectors & shortcuts
      case Expression::t_and:
      case Expression::t_or:
      {
         xValue = l_value;

         String opname, ifmode;
         if ( exp->type() == Expression::t_or )
         {
            opname = "OR  ";
            ifmode = "IFT ";
         }
         else {
            opname = "AND ";
            ifmode = "IFF ";
         }

         if( exp->first()->isSimple() && exp->second()->isSimple() )
         {
            m_out->writeString( "\t" + opname );
            gen_operand( exp->first() );
            m_out->writeString( ", " );
            gen_operand( exp->second() );
            m_out->writeString( "\n" );
         }
         else if( exp->first()->isSimple() )
         {
            int branch = m_branch_id++;
            String branchStr;
            branchStr.writeNumber( (int64) branch );

            m_out->writeString( "\tSTO \tA, " );
            gen_operand(  exp->first() );
            m_out->writeString( "\n" );
            m_out->writeString( "\t" + ifmode + "\t_branch_orand_" );
            m_out->writeString( branchStr + ", A\n" );
            // if A that is the first op is false, we generate this one.
            gen_value(  exp->second() );
            m_out->writeString( "_branch_orand_" + branchStr + ":\n" );
         }
         else if( exp->second()->isSimple() )
         {
            int branch = m_branch_id++;
            String branchStr;
            branchStr.writeNumber( (int64) branch );
            gen_value(  exp->first() );
            m_out->writeString( "\t" + ifmode + "\t_branch_orand_" + branchStr + ", A" + "\n" );
            // else we have to load the second in A
            m_out->writeString( "\tSTO \tA, " );
            gen_operand(  exp->second() );
            m_out->writeString( "\n" );
            m_out->writeString( "_branch_orand_" + branchStr + ":\n" );
         }
         else
         {
            int branch = m_branch_id++;
            String branchStr;
            branchStr.writeNumber( (int64) branch );
            gen_value(  exp->first() );
            m_out->writeString( "\t" + ifmode + "\t_branch_orand_");
            m_out->writeString( branchStr + ", A\n" );
            // else we have to load the second in A
            gen_value( exp->second() );
            m_out->writeString( "_branch_orand_" + branchStr + ":\n" );
         }
      }
      return;


      // unary operators
      case Expression::t_neg: mode = 1; opname = "NEG "; break;
      case Expression::t_not: mode = 1; opname = "NOT "; break;
      case Expression::t_bin_not: mode = 1; opname = "BNOT"; break;
      case Expression::t_strexpand: mode = 1; opname = "STEX"; break;
      case Expression::t_indirect: mode = 1; opname = "INDI"; break;
      case Expression::t_eval: mode = 1; opname = "EVAL"; break;

      case Expression::t_bin_and: mode = 2; opname = "BAND"; break;
      case Expression::t_bin_or: mode = 2; opname = "BOR "; break;
      case Expression::t_bin_xor: mode = 2; opname = "BXOR"; break;
      case Expression::t_shift_left: mode = 2; opname = "SHL "; break;
      case Expression::t_shift_right: mode = 2; opname = "SHR "; break;

      case Expression::t_plus: mode = 2; opname = "ADD "; break;
      case Expression::t_minus: mode = 2; opname = "SUB "; break;
      case Expression::t_times: mode = 2; opname = "MUL "; break;
      case Expression::t_divide: mode = 2;opname = "DIV "; break;
      case Expression::t_modulo: mode = 2; opname = "MOD "; break;
      case Expression::t_power: mode = 2; opname = "POW "; break;

      case Expression::t_gt: mode = 2; opname = "GT  "; break;
      case Expression::t_ge: mode = 2; opname = "GE  "; break;
      case Expression::t_lt: mode = 2; opname = "LT  "; break;
      case Expression::t_le: mode = 2; opname = "LE  "; break;
      case Expression::t_eq: mode = 2; opname = "EQ  "; break;
      case Expression::t_exeq: mode = 2; opname = "EXEQ"; break;
      case Expression::t_neq: mode = 2; opname = "NEQ "; break;

      case Expression::t_has: mode = 2; opname = "HAS "; break;
      case Expression::t_hasnt: mode = 2; opname = "HASN"; break;
      case Expression::t_in: mode = 2; opname = "IN  "; break;
      case Expression::t_notin: mode = 2; opname = "NOIN"; break;
      case Expression::t_provides: mode = 2; opname = "PROV"; break;

      // it is better to handle the rest directly here.
      case Expression::t_iif:
      {
         xValue = l_value;

         int32 branch = m_branch_id++;
         String branchStr;
         branchStr.writeNumber( (int64) branch );
         // condition
         if ( exp->first()->isSimple() )
         {
            m_out->writeString( "\tIFF \t_iif_fail_" + branchStr + ", " );
            gen_operand( exp->first() );
            m_out->writeString( "\n" );
         }
         else {
            gen_value( exp->first() );
            m_out->writeString( "\tIFF \t_iif_fail_" + branchStr + ", A\n" );
         }

         // success case
         if( exp->second()->isSimple() )
         {
            m_out->writeString( "\tSTO  \tA, " );
            gen_operand( exp->second() );
            m_out->writeString( "\n" );
         }
         else
         {
            gen_value( exp->second() );
         }

         m_out->writeString( "\tJMP \t_iif_end_" + branchStr + "\n" );
         m_out->writeString( "_iif_fail_" + branchStr + ":\n" );

         //failure case
         // success case
         if( exp->third()->isSimple() )
         {
            m_out->writeString( "\tSTO  \tA, " );
            gen_operand( exp->third() );
            m_out->writeString( "\n" );
         }
         else
         {
            gen_value( exp->third() );
         }

         m_out->writeString( "_iif_end_" + branchStr + ":\n" );

      }
      return;

      case Expression::t_fbind: mode = 2; opname = "FORB"; break;

      case Expression::t_assign:
         // handle it as a load...
         // don't change x-value
         gen_load( exp->first(), exp->second() );
         return;

      case Expression::t_aadd:
         xValue = l_value;
         gen_autoassign( "ADDS", exp->first(), exp->second() );
         return;

      case Expression::t_asub:
         xValue = l_value;
         gen_autoassign( "SUBS", exp->first(), exp->second() );
         return;

      case Expression::t_amul:
         xValue = l_value;
         gen_autoassign( "MULS", exp->first(), exp->second() );
         return;

      case Expression::t_adiv:
         xValue = l_value;
         gen_autoassign( "DIVS", exp->first(), exp->second() );
         return;

      case Expression::t_amod:
         xValue = l_value;
         gen_autoassign( "MODS", exp->first(), exp->second() );
         return;

      case Expression::t_apow:
         xValue = l_value;
         gen_autoassign( "POWS", exp->first(), exp->second() );
         return;

      case Expression::t_aband:
         xValue = l_value;
         gen_autoassign( "ANDS", exp->first(), exp->second() );
         return;

      case Expression::t_abor:
         xValue = l_value;
         gen_autoassign( "ORS", exp->first(), exp->second() );
         return;

      case Expression::t_abxor:
         xValue = l_value;
         gen_autoassign( "XORS", exp->first(), exp->second() );
         return;

      case Expression::t_ashl:
         xValue = l_value;
         gen_autoassign( "SHLS", exp->first(), exp->second() );
         return;

      case Expression::t_ashr:
         xValue = l_value;
         gen_autoassign( "SHRS", exp->first(), exp->second() );
         return;

      case Expression::t_pre_inc: xValue = l_value; gen_inc_prefix( exp->first() ); return;
      case Expression::t_pre_dec: xValue = l_value; gen_dec_prefix( exp->first() ); return;
      case Expression::t_post_inc: xValue = l_value; gen_inc_postfix( exp->first() ); return;
      case Expression::t_post_dec: xValue = l_value; gen_dec_postfix( exp->first() ); return;

      case Expression::t_obj_access:
         xValue = p_value;
         gen_load_from_deep( "LDP ", exp->first(), exp->second() );
      return;

      case Expression::t_array_access:
         xValue = v_value;
         gen_load_from_deep( "LDV ", exp->first(), exp->second() );
      return;

      case Expression::t_array_byte_access:
         xValue = l_value;
         gen_load_from_deep( "LSB ", exp->first(), exp->second() );
      return;

      case Expression::t_funcall:
      case Expression::t_inherit:
      {
         xValue = l_value;
         gen_funcall( exp, false );
      }
      // funcall is complete here
      return;

      case Expression::t_lambda:
      {
         xValue = l_value;
         if( exp->second() != 0 )
         {
            // we must create the lambda closure
            int size = 0;
            ListElement *iter = exp->second()->asArray()->begin();
            while( iter != 0 )
            {
               const Value *val = (Value *) iter->data();
               if( val->isSimple() ) {
                  m_out->writeString( "\tPSHR\t" );
                  gen_operand( val );
               }
               else {
                  gen_complex_value( val );
                  m_out->writeString( "\tPSHR\tA" );
               }
               m_out->writeString( "\n" );
               
               size++;
               iter = iter->next();
            }

            String temp; temp.writeNumber((int64) size);
            m_out->writeString( "\tCLOS\t" + temp +
                  ", A, $" + exp->first()->asSymbol()->name() + "\n" );
         }
         else
         {
            m_out->writeString( "\tSTO \tA, $" + exp->first()->asSymbol()->name() + "\n" );

         }
      }
      return;

      case Expression::t_oob:
         xValue = l_value;
         m_out->writeString( "\tOOB  \t1, " );
            gen_operand( exp->first() );
            m_out->writeString( "\n" );
      return;

      case Expression::t_deoob:
         xValue = l_value;
         m_out->writeString( "\tOOB  \t0, " );
            gen_operand( exp->first() );
            m_out->writeString( "\n" );
      return;

      case Expression::t_xoroob:
         xValue = l_value;
         m_out->writeString( "\tOOB  \t2, " );
            gen_operand( exp->first() );
            m_out->writeString( "\n" );
      return;

      case Expression::t_isoob:
         xValue = l_value;
         m_out->writeString( "\tOOB  \t3, " );
            gen_operand( exp->first() );
            m_out->writeString( "\n" );
      return;
   }

   // post-processing unary and binary operators.
   // ++, -- and accessors are gone; safely change the l-value status.
   xValue = l_value;

   // then, if there is still something to do, put the operands in place.
   if ( mode == 1 ) {  // unary?
      if ( exp->first()->isSimple() ) {
         m_out->writeString( "\t" + opname + "\t" );
         gen_operand( exp->first() );
      }
      else {
         gen_complex_value( exp->first() );
         m_out->writeString( "\t" + opname + "\tA" );
      }
   }
   else {
      if ( exp->first()->isSimple() && exp->second()->isSimple() ) {
         m_out->writeString( "\t" + opname + "\t" );
         gen_operand( exp->first() );
         m_out->writeString( ", " );
         gen_operand( exp->second() );
      }
      else if ( exp->first()->isSimple() ) {
         gen_complex_value( exp->second() );
         m_out->writeString( "\t" + opname + "\t" );
         gen_operand( exp->first() );
         m_out->writeString( ", A" );
      }
      else if ( exp->second()->isSimple() ) {
         gen_complex_value( exp->first() );
         m_out->writeString( "\t" + opname + "\tA, " );
         gen_operand( exp->second() );
      }
      else {
         gen_complex_value( exp->first() );
         m_out->writeString( "\tPUSH\tA\n" );
         gen_value( exp->second() );
         m_out->writeString( "\tPOP \tB\n" );
         m_out->writeString( "\t" + opname + "\tB, A" );
      }
   }

   m_out->writeString( "\n" );
}

void GenHAsm::gen_dict_decl( const DictDecl *dcl )
{
   int size = 0;

   ListElement *iter = dcl->begin();
   while( iter != 0 )
   {
      DictDecl::pair *pair = (DictDecl::pair *) iter->data();
      const Value *key = pair->first;
      const Value *value = pair->second;

      gen_push( key );
      gen_push( value );
      size++;
      iter = iter->next();
   }
   String sizeStr;
   sizeStr.writeNumber( (int64) size );

   m_out->writeString( "\tGEND\t" + sizeStr + "\n" );
}

void GenHAsm::gen_array_decl( const ArrayDecl *dcl )
{
   int size = 0;

   ListElement *iter = dcl->begin();
   while( iter != 0 )
   {
      const Value *val = (const Value *) iter->data();
      gen_push( val );
      size++;
      iter = iter->next();
   }
   String sizeStr;
   sizeStr.writeNumber( (int64) size );
   m_out->writeString( "\tGENA\t" + sizeStr + "\n" );
}


void GenHAsm::gen_range_decl( const RangeDecl *dcl )
{
   if ( dcl->isOpen() )
   {
      if ( dcl->rangeStart()->isSimple() )
      {
         m_out->writeString( "\tGEOR\t" );
         gen_operand( dcl->rangeStart() );
         m_out->writeString( "\n" );
      }
      else {
         gen_complex_value( dcl->rangeStart() );
         m_out->writeString( "\tGEOR\tA\n" );
      }
   }
   else
   {
      Value dummy; // defaults to nil
      Value *rangeStep;
      if ( dcl->rangeStep() == 0 )
      {
         dummy.setInteger( 0 );
         rangeStep = &dummy;
      }
      else if ( dcl->rangeStep()->isSimple() )
      {
         rangeStep = dcl->rangeStep();
      }
      else
      {
         gen_complex_value( dcl->rangeStep() );
         m_out->writeString( "\tPUSH\tA\n" );
         // we'll instruct GENR to get it via NIL as parameter
         rangeStep = &dummy;
      }

      if ( dcl->rangeStart()->isSimple() && dcl->rangeEnd()->isSimple() )
      {
         m_out->writeString( "\tGENR\t" );
         gen_operand( dcl->rangeStart() );
         m_out->writeString( ", " );
         gen_operand( dcl->rangeEnd() );
         m_out->writeString( ", " );
         gen_operand( rangeStep );
         m_out->writeString( "\n" );
      }
      else if ( dcl->rangeStart()->isSimple() )
      {
         gen_complex_value( dcl->rangeEnd() );
         m_out->writeString( "\tGENR\t" );
         gen_operand( dcl->rangeStart() );
         m_out->writeString( ", A, " );
         gen_operand( rangeStep );
         m_out->writeString( "\n" );
      }
      else if ( dcl->rangeEnd()->isSimple() )
      {
         gen_complex_value( dcl->rangeStart() );
         m_out->writeString( "\tGENR\tA, " );
         gen_operand( dcl->rangeEnd() );
         m_out->writeString( ", " );
         gen_operand( rangeStep );
         m_out->writeString( "\n" );
      }
      else {
         gen_complex_value( dcl->rangeStart() );
         m_out->writeString( "\tPUSH\tA\n" );
         gen_complex_value( dcl->rangeEnd() );
         m_out->writeString( "\tPOP \tB\n" );
         m_out->writeString( "\tGENR\tB, A, ");
         gen_operand( rangeStep );
         m_out->writeString( "\n" );
      }
   }
}

void GenHAsm::gen_load( const Value *target, const Value *source )
{
   if ( target->isSimple() && source->isSimple() )
   {
      m_out->writeString( "\tLD  \t" );
      gen_operand( target );
      m_out->writeString( ", " );
      gen_operand( source );
      m_out->writeString( "\n" );
   }
   else if ( target->isSimple() )
   {
      if( source->isReference() )
      {
         if ( source->asReference() == 0 )
         {
            m_out->writeString( "\tLDRF\t" );
            gen_operand( target );
            m_out->writeString( ", 0\n" );
         }
         else {
            if( source->asReference()->isSymbol() )
            {
               m_out->writeString( "\tLDRF\t" );
               gen_operand( target );
               m_out->writeString( ", $" + source->asReference()->asSymbol()->name() + "\n" );
            }
            else {
               gen_value( source->asReference() );
               m_out->writeString( "\tLDRF\t" );
               gen_operand( target );
               m_out->writeString( ", A\n" );
            }
         }
      }
      else {
         gen_complex_value( source );
         m_out->writeString( "\tLD  \t" );
         gen_operand( target );
         m_out->writeString( ", A\n" );
      }
   }
   else {
      // target is NOT simple. If it's an expression ...
      if( target->type() == Value::t_expression )
      {
         const Expression *exp = target->asExpr();
         // ... then it may be an array assignment...

         if( exp->type() == Expression::t_array_access ) {
            gen_store_to_deep( "STV", exp->first(), exp->second(), source );
         }
         else if ( exp->type() == Expression::t_obj_access ) {
            gen_store_to_deep( "STP", exp->first(), exp->second(), source );
         }
      }
      else if ( target->type() == Value::t_array_decl )
      {
         const ArrayDecl *tarr = target->asArray();

         // if the source is also an array, fine, we have a 1:1 assignment.
         if ( source->type() == Value::t_array_decl ) {
            const ArrayDecl *sarr = source->asArray();

            ListElement *it_s = sarr->begin();
            ListElement *it_t = tarr->begin();

            while( it_s != 0 && it_t != 0 ) {
               const Value *t = (const Value *) it_t->data();
               const Value *s = (const Value *) it_s->data();
               gen_load( t, s );
               it_s = it_s->next();
               it_t = it_t->next();
            }
            // the compiler takes care to provide us with parallel arrays,
            // but we set a trap here in case there is a bug in the compiler.
            fassert( it_s == 0 && it_t == 0 );
         }
         // we must generate an unpack request
         else
         {
            String ssize;
            ssize.writeNumber( (int64) tarr->size() );
            // then unpack the source in the array.
            if ( source->isSimple() )
            {
               m_out->writeString( "\tLDAS\t" + ssize + ", " );
               gen_operand( source );
               m_out->writeString( "\n" );
            }
            else {
					gen_complex_value( source );
					m_out->writeString( "\tLDAS\t" + ssize + ", A\n" );
            }

            ListElement *it_t = tarr->end();
            while( it_t != 0 ) {
               const Value *t = (const Value *) it_t->data();
               if( t->isSimple() )
               {
                  m_out->writeString( "\tPOP \t" );
                  gen_operand( t );
                  m_out->writeString( "\n" );
               }
               else {
                  m_out->writeString( "\tPOP \tB\n" );
                  gen_load_from_reg( t, "B" );
               }
               it_t = it_t->prev();
            }
         }
      }
   }
}

int GenHAsm::gen_refArray( const ArrayDecl *tarr, bool bGenArray )
{
   ListElement *it_t = tarr->begin();
   int size = 0;

   // first generates an array of references
   while( it_t != 0 ) {
      // again, is the compiler that must make sure of this...
      const Value *val = (const Value *) it_t->data();
      fassert( val->isSimple() );

      m_out->writeString( "\tPSHR\t" );
      gen_operand( val );
      m_out->writeString( "\n" );

      ++size;
      it_t = it_t->next();
   }
   String sizeStr;
   sizeStr.writeNumber( (int64) size );

	if ( bGenArray )
		m_out->writeString( "\tGENA\t" + sizeStr + "\n" );
   return size;
}

void GenHAsm::gen_store_to_deep( const char *type, const Value *first, const Value *second, const Value *source )
{
   String operation;
   String typeStr = type;

   // first we must generate the assignands.
   if( source->isSimple() )
   {
      if ( first->isSimple() && second->isSimple() ) {
         m_out->writeString( "\t" + typeStr + " \t" );
         gen_operand( first );
         m_out->writeString( ", " );
         gen_operand( second );
         m_out->writeString( ", " );
         gen_operand( source );
         m_out->writeString( "\n" );
      }
      else if ( second->isSimple() ) {
         gen_complex_value( first );
         m_out->writeString( "\t" + typeStr + " \tA, " );
         gen_operand( second );
         m_out->writeString( ", " );
         gen_operand( source );
         m_out->writeString( "\n" );
      }
      else if ( first->isSimple() ) {
         gen_complex_value( second );
         m_out->writeString( "\t" + typeStr + " \t" );
         gen_operand( first );
         m_out->writeString( ",A , " );
         gen_operand( source );
         m_out->writeString( "\n" );
      }
      else {
         gen_complex_value( first );
         m_out->writeString( "\tPUSH\tA\n" );
         gen_complex_value( second );
         m_out->writeString( "\tPOP \tB\n" );
         m_out->writeString( "\t" +  typeStr + " \tB, A, " );
         gen_operand( source );
         m_out->writeString( "\n" );
      }
   }
   else {
      if ( first->isSimple() && second->isSimple() )
      {
         gen_complex_value( source );
         m_out->writeString( "\t" + typeStr + " \t" );
         gen_operand( first );
         m_out->writeString( ", " );
         gen_operand( second );
         m_out->writeString( ", A\n" );
      }
      else {
         if( source->isReference() ) {
            operation = "R"; // store to vector by reference
            source = source->asReference();
         }
         else {
            operation = "S"; // store to vector from stack top
         }

         if ( second->isSimple() ) {
            gen_complex_value( first );
            m_out->writeString( "\tPUSH\tA\n" );
            gen_complex_value( source );
            m_out->writeString( "\tXPOP\tA\n" );
            m_out->writeString( "\t" + typeStr + operation + "\tA, " );
            gen_operand( second );
            m_out->writeString( "\n" );
         }
         else if ( first->isSimple() ) {
            gen_complex_value( second );
            m_out->writeString( "\tPUSH\tA\n" );
            gen_complex_value( source );
            m_out->writeString( "\tXPOP\tA\n" );
            m_out->writeString( "\t" + typeStr + operation + "\t" );
            gen_operand( first );
            m_out->writeString( ", A\n" );
         }
         else {
            gen_complex_value( first );
            m_out->writeString( "\tPUSH\tA\n" );
            gen_complex_value( second );
            m_out->writeString( "\tPUSH\tA\n" );
            gen_complex_value( source );
            m_out->writeString( "\tPOP \tB\n" );
            m_out->writeString( "\tXPOP\tA\n" );
            m_out->writeString( "\t" + typeStr + operation + "\tA, B\n" );
         }
      }
   }
}


void GenHAsm::gen_load_from_deep( const char *type, const Value *first, const Value *second )
{
   String typeStr = type;

   // first we must generate the assignands.
   if( first->isSimple() )
   {
      if ( second->isSimple() ) {
         m_out->writeString( "\t" + typeStr + "\t" );
         gen_operand( first );
         m_out->writeString( ", " );
         gen_operand( second );
         m_out->writeString( "\n" );
      }
      else {
         gen_complex_value( second );
         m_out->writeString( "\t" + typeStr + "\t" );
         gen_operand( first );
         m_out->writeString( ", A\n" );
      }
   }
   else {
      if ( second->isSimple() )
      {
         gen_complex_value( first );
         m_out->writeString( "\t" + typeStr + " \t" );
         m_out->writeString( "A, " );
         gen_operand( second );
         m_out->writeString( "\n" );
      }
      else {
         gen_complex_value( first );
         m_out->writeString( "\tPUSH\tA\n" );
         gen_complex_value( second );
         m_out->writeString( "\tPOP \tB\n" );
         m_out->writeString( "\t" + typeStr + "\tB, A\n" );
      }
   }
}



void GenHAsm::gen_load_from_A( const Value *target )
{
   gen_load_from_reg( target, "A" );
}

void GenHAsm::gen_load_from_reg( const Value *target, const char *reg )
{
   String regStr = reg;

   if ( target->isSimple() )
   {
      m_out->writeString( "\tLD  \t" );
      gen_operand( target );
      m_out->writeString( ", " + regStr + "\n" );
   }
   else {
      // target is NOT simple. If it's an expression ...
      if( target->type() == Value::t_expression )
      {
         const Expression *exp = target->asExpr();
         // ... then it may be an array assignment...

         if( exp->type() == Expression::t_array_access ) {
            gen_store_to_deep_reg( "STV", exp->first(), exp->second(), reg );
         }
         else if ( exp->type() == Expression::t_obj_access ) {
            gen_store_to_deep_reg( "STP", exp->first(), exp->second(), reg );
         }
         else {
            m_out->writeString( "\tPUSH\t" + regStr + "\n" );
            gen_expression( exp );
            m_out->writeString( "\tPOP \tB\n" );
            m_out->writeString( "\tLD  \tA,B\n" );
         }
      }
      else if ( target->type() == Value::t_array_decl )
      {
         // if the source is also an array, fine, we have a 1:1 assignment.
         const ArrayDecl *tarr = target->asArray();
         String ssize;
         ssize.writeNumber((int64) tarr->size() );

         // push the source array on the stack.
         m_out->writeString( "\tLDAS\t" + ssize + ", " + regStr + "\n" );

         // and load each element back in the array
         ListElement *it_t = tarr->end();

         // Now load each element by popping it.
         while( it_t != 0 )
         {
            const Value *val = (const Value *) it_t->data();
            if( val->isSimple() )
            {
               m_out->writeString( "\tPOP \t" );
               gen_operand( val );
               m_out->writeString( "\n" );
            }
            else {
               m_out->writeString( "\tPOP \tB\n" );
               gen_load_from_reg( val, "B" );
            }
            it_t = it_t->prev();
         }
      }
   }
}

void GenHAsm::gen_store_to_deep_A( const char *type, const Value *first, const Value *second )
{
   gen_store_to_deep_reg( type, first, second, "A" );
}

void GenHAsm::gen_store_to_deep_reg( const char *type, const Value *first, const Value *second, const char *reg )
{
   String typeStr = type;

   // first we must generate the assignands.
   if ( first->isSimple() && second->isSimple() ) {
      m_out->writeString( "\t" + typeStr + " \t" );
      gen_operand( first );
      m_out->writeString( ", " );
      gen_operand( second );
      m_out->writeString( ", " );
      m_out->writeString( reg );
      m_out->writeString( "\n" );
   }
   else {

      m_out->writeString( "\tPUSH\t" + typeStr + "\n" );
      if ( second->isSimple() ) {
         gen_complex_value( first );
         m_out->writeString( "\t" + typeStr + "S\tA, " );
         gen_operand( second );
         m_out->writeString( "\n" );
      }
      else if ( first->isSimple() ) {
         gen_complex_value( second );
         m_out->writeString( "\t" + typeStr + "S\t" );
         gen_operand( first );
         m_out->writeString( ", A\n" );
      }
      else {
         gen_complex_value( first );
         m_out->writeString( "\tPUSH\tA\n" );
         gen_complex_value( second );
         m_out->writeString( "\tPOP \tB\n" );
         m_out->writeString( "\t" + typeStr + "S\tB, A\n" );
      }
   }
}


void GenHAsm::gen_funcall( const Expression *exp, bool fork )
{
   int size = 0;
   int branch;

   String functor = exp->type() == Expression::t_inherit ? "INST" : "CALL";

   if( exp->second() != 0 )
   {
      const ArrayDecl *dcl = exp->second()->asArray();
      ListElement *iter = dcl->begin();
      while( iter != 0 )
      {
         const Value *val = (const Value *) iter->data();
         gen_push( val );
         size++;
         iter = iter->next();
      }
   }
   String sizeStr;
   sizeStr.writeNumber( (int64) size );

   if ( fork ) {
      branch = m_branch_id++;
      String branchStr;
      branchStr.writeNumber( (int64) branch );
      if( exp->first()->isSimple() ) {
         m_out->writeString( "\tPUSH\t" );
         gen_operand( exp->first() );
         m_out->writeString( "\n" );
      }
      else {
         gen_complex_value( exp->first() );
         m_out->writeString( "\tPUSH\t A\n" );
      }
      String sizeOne;
      sizeOne.writeNumber( (int64) size + 1 );
      m_out->writeString( "\tFORK\t" + sizeOne + ", _branch_fork_" );
      m_out->writeString( branchStr + "\n" );
      m_out->writeString( "\tPOP \tA\n" );
      m_out->writeString( "\t" + functor + "\t" );
      m_out->writeString( sizeStr + ", A\n" );
      m_out->writeString( "\tEND \t\n" );
      m_out->writeString( "_branch_fork_" + branchStr + ":\n" );
   }
   else {
      if( exp->first()->isSimple() ) {
         m_out->writeString( "\t" + functor + "\t" );
         m_out->writeString( sizeStr + ", " );
         gen_operand( exp->first() );
         m_out->writeString( "\n" );
      }
      else {
         gen_complex_value( exp->first() );
         m_out->writeString( "\t" + functor + "\t" );
         m_out->writeString( sizeStr + ", A\n" );
      }
   }
}

}

/* end of genhasm.cpp */
