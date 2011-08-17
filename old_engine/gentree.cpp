/*
   FALCON - The Falcon Programming Language.
   FILE: gentree.cpp

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: $DATE
   Last modified because

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/gentree.h>
#include <falcon/syntree.h>
#include <falcon/compiler.h>
#include <falcon/stream.h>

namespace Falcon
{

void GenTree::generate( const SourceTree *st )
{
   // generate functions
   if ( ! st->functions().empty() ) {
      m_out->writeString( "FUNCTIONS: \n" );

      const StmtFunction *func = static_cast<const StmtFunction*>( st->functions().front() );
      while ( func != 0 )
      {
         m_out->writeString( "Function " + func->name() + "(" );
         const Symbol *fsym = func->symbol();
         const FuncDef *fdef = fsym->getFuncDef();

         if( fdef->params() != 0)
         {
            MapIterator iter = fdef->symtab().map().begin();
            bool first = true;

            while( iter.hasCurrent() )
            {
               const Symbol *param = *(const Symbol **) iter.currentValue();
               if( param->isParam() ) {
                  if ( first ) {
                     first = false;
                     m_out->writeString( " " );
                  }
                  else
                     m_out->writeString( ", " );

                  m_out->writeString( param->name() );
               }

               iter.next();
            }
         }
         m_out->writeString( " )\n" );

         if( func->hasStatic() ) {
            m_out->writeString( "STATIC:\n" );
            gen_block( func->staticBlock(), 1 );
         }

         gen_block( func->statements(), 1 );

         func = static_cast<const StmtFunction *>(func->next());
      }
      m_out->writeString( "\n" );
   }

   m_out->writeString( "CODE:\n" );
   gen_block( st->statements(), 0 );
}

void GenTree::generate( const Statement *cmp, const char *specifier, bool sameline, int depth )
{

   if ( ! sameline ) {
      String line;
      line.writeNumber( (int64) cmp->line() );
      int pos = 0;
      while (pos + line.length() < 5 ) {
         pos ++;
         m_out->writeString( " " );
      }

      m_out->writeString( line + " : " );

      for (int i = 0; i < depth; i++ )
         m_out->writeString( " " );
   }

   if ( specifier != 0 ) {
      m_out->writeString( specifier );
      m_out->writeString( " " );
   }

   switch( cmp->type() )
   {
      case Statement::t_none: m_out->writeString( "(placeholder none statement)\n" ); break;
      case Statement::t_break: m_out->writeString( "BREAK\n" ); break;
      case Statement::t_continue:
         m_out->writeString( "CONTINUE" );
         if( static_cast< const StmtContinue *>( cmp )->dropping() )
            m_out->writeString( " DROPPING" );
         m_out->writeString( "\n" );
         break;

      case Statement::t_launch:
         m_out->writeString( "LAUNCH " );
         gen_value( static_cast< const StmtExpression *>( cmp )->value() );
         m_out->writeString( "\n" );
      break;

      case Statement::t_autoexp:
         m_out->writeString( "AUTOEXPR " );
         gen_value( static_cast< const StmtExpression *>( cmp )->value() );
         m_out->writeString( "\n" );
      break;

      case Statement::t_return:
         m_out->writeString( "RETURN " );
         gen_value( static_cast< const StmtExpression *>( cmp )->value() );
         m_out->writeString( "\n" );
      break;

      case Statement::t_fordot:
         m_out->writeString( "FORDOT " );
         gen_value( static_cast< const StmtExpression *>( cmp )->value() );
         m_out->writeString( "\n" );
      break;

      case Statement::t_raise:
         m_out->writeString( "RAISE " );
         gen_value( static_cast< const StmtExpression *>( cmp )->value() );
         m_out->writeString( "\n" );
      break;

      case Statement::t_give:
      {
         const StmtGive *give = static_cast< const StmtGive *>( cmp );
         m_out->writeString( "GIVE " );
         gen_array( give->attributes() );
         m_out->writeString( " to " );
         gen_array( give->objects() );
         m_out->writeString( "\n" );
      }
      break;

      case Statement::t_self_print:
      {
         const StmtSelfPrint *sp = static_cast< const StmtSelfPrint *>( cmp );

         m_out->writeString( "FAST PRINT " );

         gen_array( sp->toPrint() );
         m_out->writeString( "\n" );
      }
      break;

      case Statement::t_if:
      {
         m_out->writeString( "IF " );
         const StmtIf *sif = static_cast< const StmtIf *>( cmp );
         gen_value( sif->condition() );
         m_out->writeString( "\n" );
         gen_block( sif->children(), depth );
         const Statement *stmt = sif->elifChildren().front();
         while( stmt != 0 ) {
            generate( stmt, 0, false, depth + 1 );
            stmt = static_cast<const Statement *>(stmt->next());
         }
         gen_block( sif->elseChildren(), depth, "ELSE" );
      }
      break;

      case Statement::t_select:
      case Statement::t_switch:
      {
         if ( cmp->type() == Statement::t_switch )
            m_out->writeString( "SWITCH " );
         else
            m_out->writeString( "SELECT " );

         const StmtSwitch *sw = static_cast< const StmtSwitch *>( cmp );
         gen_value( sw->switchItem() );
         m_out->writeString( "\n" );

         // generates the switch lists
         MapIterator iter;

         // generatest the switch integer list
         if ( !sw->intCases().empty() )
         {
            m_out->writeString( "INT cases " );
            iter = sw->intCases().begin();
            while( iter.hasCurrent() )
            {
               Value *first = *(Value **) iter.currentKey();
               uint32 second = *(uint32 *) iter.currentValue();
               String temp;
               temp.writeNumber( (int64) second );

               gen_value( first );
               m_out->writeString( "->" + temp + "; " );
               iter.next();
            }
            m_out->writeString( "\n" );
         }

         if ( !sw->rngCases().empty() )
         {
            m_out->writeString( "RANGE cases " );
            iter = sw->rngCases().begin();
            while( iter.hasCurrent() )
            {
               Value *first = *(Value **) iter.currentKey();
               uint32 second = *(uint32 *) iter.currentValue();
               String temp;
               temp.writeNumber( (int64) second );

               gen_value( first );
               m_out->writeString( "->" + temp + "; " );
               iter.next();
            }
            m_out->writeString( "\n" );
         }

         if ( !sw->strCases().empty() )
         {
            m_out->writeString( "STRING cases " );
            iter = sw->strCases().begin();
            while( iter.hasCurrent() )
            {
               Value *first = *(Value **) iter.currentKey();
               uint32 second = *(uint32 *) iter.currentValue();
               String temp;
               temp.writeNumber( (int64) second );

               gen_value( first );
               m_out->writeString( "->" + temp + "; " );
               iter.next();
            }
            m_out->writeString( "\n" );
         }

         if ( !sw->objCases().empty() )
         {
            m_out->writeString( "Symbol cases " );
            iter = sw->objCases().begin();
            while( iter.hasCurrent() )
            {
               Value *first = *(Value **) iter.currentKey();
               uint32 second = *(uint32 *) iter.currentValue();
               String temp;
               temp.writeNumber( (int64) second );

               gen_value( first );
               m_out->writeString( "->" + temp + "; " );
               iter.next();
            }
            m_out->writeString( "\n" );
         }


         // generates the blocks
         int blockId = 0;
         const Statement *stmt = sw->blocks().front();
         while( stmt != 0 ) {
            String blockStr;
            blockStr.writeNumber( (int64) blockId );
            if( blockId == sw->nilBlock() )
               m_out->writeString( "CASE BLOCK (NIL)" + blockStr + "\n" );
            else
               m_out->writeString( "CASE BLOCK " + blockStr + "\n" );

            generate( stmt, 0, false, depth + 1 );
            stmt = static_cast<const Statement *>(stmt->next());
            blockId ++ ;
         }
         if ( ! sw->defaultBlock().empty() )
         {
            m_out->writeString( "DEFAULT BLOCK\n" );
            gen_block( sw->defaultBlock(), depth + 1 );
         }

      }
      break;

      case Statement::t_case:
      {
         //m_out->writeString( "CASE \n" );
         const StmtCaseBlock *scase = static_cast< const StmtCaseBlock *>( cmp );
         gen_block( scase->children(), depth );
      }
      break;

      case Statement::t_catch:
      {
         //m_out->writeString( "CASE \n" );
         const StmtCatchBlock *scase = static_cast< const StmtCatchBlock *>( cmp );
         if ( scase->intoValue() != 0 )
         {
            m_out->writeString( "CATCH into " );
            gen_value( scase->intoValue() );
            m_out->writeString( "\n" );
         }
         else
            m_out->writeString( "CATCH witout into\n" );

         gen_block( scase->children(), depth );
      }
      break;

      case Statement::t_elif:
      {
         m_out->writeString( "ELIF " );
         const StmtElif *selif = static_cast< const StmtElif *>( cmp );
         gen_value( selif->condition() );
         m_out->writeString( "\n" );
         gen_block( selif->children(), depth );
      }
      break;

      case Statement::t_while:
      {

         const StmtWhile *wh = static_cast< const StmtWhile *>( cmp );
         m_out->writeString( "WHILE " );
         gen_value( wh->condition() );

         m_out->writeString( "\n" );
         gen_block( wh->children(), depth );
      }
      break;

      case Statement::t_loop:
      {
         const StmtLoop *wh = static_cast< const StmtLoop *>( cmp );
         m_out->writeString( "LOOP " );
         m_out->writeString( "\n" );
         gen_block( wh->children(), depth );

         if( wh->condition() != 0 )
         {
            m_out->writeString( "END LOOP WHEN " );
            gen_value( wh->condition() );
            m_out->writeString( "\n" );
         }
         else
            m_out->writeString( "END\n" );
      }
      break;

      case Statement::t_global:
      {
         m_out->writeString( "GLOBAL " );
         const StmtGlobal *sglobal = static_cast< const StmtGlobal *>( cmp );
         ListElement *iter = sglobal->getSymbols().begin();
         while ( iter != 0 ) {
            Symbol *sym = (Symbol *) iter->data();
            m_out->writeString( sym->name() + ", " );
            iter = iter->next();
         }
         m_out->writeString( "\n" );
      }
      break;

      case Statement::t_forin:
      {
         m_out->writeString( "FOR-IN " );
         const StmtForin *sfor = static_cast< const StmtForin *>( cmp );
         gen_array( sfor->dest() );
         m_out->writeString( " IN " );
         gen_value( sfor->source() );
         m_out->writeString( "\n" );
         gen_block( sfor->children(), depth );
         gen_block( sfor->firstBlock(), depth, "FORFIRST" );
         gen_block( sfor->middleBlock(), depth, "FORMIDDLE" );
         gen_block( sfor->lastBlock(), depth, "FORLAST" );
      }
      break;

      case Statement::t_try:
      {
         m_out->writeString( "TRY\n" );
         const StmtTry *stry = static_cast< const StmtTry *>( cmp );
         gen_block( stry->children(), depth );
         // generatest the switch integer list
         if ( ! stry->intCases().empty() )
         {
            m_out->writeString( "TYPE ID CATCHES " );
            MapIterator iter = stry->intCases().begin();
            while( iter.hasCurrent() )
            {
               Value *first = *(Value **) iter.currentKey();
               uint32 second = *(uint32 *) iter.currentValue();
               String temp;
               temp.writeNumber( (int64) second );

               gen_value( first );
               m_out->writeString( "->" + temp + "; " );
               iter.next();
            }
            m_out->writeString( "\n" );
         }

         // Generates the switch symbol list
         if ( ! stry->objCases().empty() )
         {
            m_out->writeString( "SYMBOL CATCHES " );
            MapIterator  iter = stry->objCases().begin();
            while( iter.hasCurrent() )
            {
               Value *first = *(Value **) iter.currentKey();
               uint32 second = *(uint32 *) iter.currentValue();
               String temp;
               temp.writeNumber( (int64) second );

               gen_value( first );
               m_out->writeString( "->" + temp + "; " );
               iter.next();
            }
            m_out->writeString( "\n" );
         }

         // generates the blocks
         int blockId = 0;
         const Statement *stmt = stry->handlers().front();
         while( stmt != 0 ) {
            String blockStr;
            blockStr.writeNumber( (int64) blockId );
            m_out->writeString( "HANDLER BLOCK " + blockStr + "\n" );

            generate( stmt, 0, false, depth + 1 );
            stmt = static_cast<const Statement *>( stmt->next() );
            blockId ++ ;
         }

         if ( stry->defaultHandler() != 0 )
         {
            m_out->writeString( "DEFAULT HANDLER" );
            if ( stry->defaultHandler()->intoValue() != 0 ) {
               m_out->writeString( " into " );
               gen_value( stry->defaultHandler()->intoValue() );
            }
            m_out->writeString( "\n" );
            gen_block( stry->defaultHandler()->children(), depth + 1 );
         }
      }
      break;

      case Statement::t_propdef:
      {
         m_out->writeString( "PROPDEF " );
         const StmtVarDef *spd = static_cast< const StmtVarDef *>( cmp );
         m_out->writeString( *spd->name() );
         m_out->writeString( "=" );
         gen_value( spd->value() );
         m_out->writeString( "\n" );
      }
      break;


      default:
         m_out->writeString( "????\n" );
   }
}

void GenTree::gen_block( const StatementList &list, int depth, const char *prefix )
{
   if( list.empty() )
      return;

   if ( prefix != 0 ) {
      String line;
      line.writeNumber( (int64) list.front()->line() );
      int pos = 0;
      while( pos + line.length() < 5 )
      {
         m_out->writeString( " " );
         pos++;
      }
      m_out->writeString( line + " : " );
      for (int i = 0; i < depth; i++ )
         m_out->writeString( " " );
      m_out->writeString( prefix );
      m_out->writeString(  "\n" );
   }

   const Statement *stmt = list.front();
   while( stmt != 0 ) {
      generate( stmt, 0, false, depth + 1 );
      stmt = static_cast<const Statement *>(stmt->next());
   }
}

void GenTree::gen_value( const Value *val )
{
   if ( val == 0 ) {
      m_out->writeString( "(null)" );
      return;
   }

   switch( val->type() )
   {
      case Value::t_nil: m_out->writeString( "nil" ); break;
      case Value::t_unbound: m_out->writeString( "unbound" ); break;
      case Value::t_imm_bool:
            m_out->writeString( val->asBool() ? "true": "false" );
      break;

      case Value::t_imm_integer: {
         String intStr;
         intStr.writeNumber( val->asInteger() );
         m_out->writeString( intStr );
      }
      break;
      case Value::t_imm_string:
      {
         String temp;
         val->asString()->escape( temp );
         m_out->writeString( "\"" + temp + "\"" );
      }
      break;

      case Value::t_lbind:
      {
         m_out->writeString( "&" + *val->asLBind() );
      }
      break;

      case Value::t_imm_num: {
         String numStr;
         numStr.writeNumber( val->asInteger() );
         m_out->writeString( numStr );
      }
      break;

      case Value::t_range_decl:
         m_out->writeString( "[" );
         gen_value( val->asRange()->rangeStart() );
         m_out->writeString( ":" );
         if ( ! val->asRange()->isOpen() )
         {
            gen_value(  val->asRange()->rangeEnd() ) ;
            if ( val->asRange()->rangeStep() != 0 )
            {
               m_out->writeString( ":" );
               gen_value(  val->asRange()->rangeStep() );
            }

         }
         m_out->writeString( "]" );
      break;

      case Value::t_symbol: m_out->writeString( val->asSymbol()->name() ); break;
      case Value::t_self: m_out->writeString( "self" ); break;
      case Value::t_array_decl:
         m_out->writeString( "Array: [ " );
            gen_array( val->asArray() );
         m_out->writeString( " ]" );
      break;

      case Value::t_dict_decl:
         m_out->writeString( "Dict: [" );
            gen_dict( val->asDict() );
         m_out->writeString( " ]" );
      break;

      case Value::t_byref:
         m_out->writeString( "$" );
         gen_value( val->asReference() );
      break;

      case Value::t_expression:
         gen_expression( val->asExpr() );
      break;

      default:
         break;
   }
}

void GenTree::gen_expression( const Expression *exp )
{

   String name;
   int type = 0;

   switch( exp->type() )
   {
      case Expression::t_optimized: gen_value( exp->first() ); return;
      case Expression::t_neg: type = 0; name = "-"; break;
      case Expression::t_bin_not: type = 0; name = "!"; break;
      case Expression::t_strexpand: type = 0; name = "@"; break;
      case Expression::t_indirect: type = 0; name = "#"; break;
      case Expression::t_not: type = 0; name = "not"; break;
      case Expression::t_pre_inc: type = 0; name = "++"; break;
      case Expression::t_pre_dec: type = 0; name = "--"; break;

      case Expression::t_eval: type = 0; name = "^*"; break;
      case Expression::t_oob: type = 0; name = "^+"; break;
      case Expression::t_deoob: type = 0; name = "^-"; break;
      case Expression::t_isoob: type = 0; name = "^?"; break;
      case Expression::t_xoroob: type = 0; name = "^!"; break;

      case Expression::t_post_inc: type = 1; name = "++"; break;
      case Expression::t_post_dec: type = 1; name = "--"; break;

      case Expression::t_bin_and: type = 2; name = "&"; break;
      case Expression::t_bin_or: type = 2; name = "|"; break;
      case Expression::t_bin_xor: type = 2; name = "^"; break;
      case Expression::t_shift_left: type = 2; name = "<<"; break;
      case Expression::t_shift_right: type = 2; name = ">>"; break;

      case Expression::t_plus: type = 2; name = "+"; break;
      case Expression::t_minus: type = 2; name = "-"; break;
      case Expression::t_times: type = 2; name = "*"; break;
      case Expression::t_divide: type = 2; name = "/"; break;
      case Expression::t_modulo: type = 2; name = "%"; break;
      case Expression::t_power: type = 2; name = "**"; break;

      case Expression::t_gt: type = 2; name = ">"; break;
      case Expression::t_ge: type = 2; name = ">="; break;
      case Expression::t_lt: type = 2; name = "<"; break;
      case Expression::t_le: type = 2; name = "<="; break;
      case Expression::t_eq: type = 2; name = "="; break;
      case Expression::t_exeq: type = 2; name = "eq"; break;
      case Expression::t_neq: type = 2; name = "!="; break;

      case Expression::t_has: type = 2; name = "has"; break;
      case Expression::t_hasnt: type = 2; name = "hasnt"; break;
      case Expression::t_in: type = 2; name = "in"; break;
      case Expression::t_notin: type = 2; name = "notin"; break;
      case Expression::t_provides: type = 2; name = "provides"; break;
      case Expression::t_or: type = 2; name = "or"; break;
      case Expression::t_and: type = 2; name = "and"; break;

      case Expression::t_iif: type = 3; break;

      case Expression::t_assign: type = 4; name = " = "; break;
      case Expression::t_fbind: type = 4; name = " | "; break;
      case Expression::t_aadd: type = 4; name = " += "; break;
      case Expression::t_asub: type = 4; name = " -= "; break;
      case Expression::t_amul: type = 4; name = " *= "; break;
      case Expression::t_adiv: type = 4; name = " /= "; break;
      case Expression::t_amod: type = 4; name = " %= "; break;
      case Expression::t_apow: type = 4; name = " *= "; break;
      case Expression::t_aband: type = 4; name = " &= "; break;
      case Expression::t_abor: type = 4; name = " |= "; break;
      case Expression::t_abxor: type = 4; name = " ^= "; break;
      case Expression::t_ashl: type = 4; name = " <<= "; break;
      case Expression::t_ashr: type = 4; name = " >>= "; break;

      case Expression::t_array_access: type = 5; break;
      case Expression::t_array_byte_access: type = 10; break;
      case Expression::t_obj_access: type = 6; break;
      case Expression::t_funcall: case Expression::t_inherit: type = 7; break;
      case Expression::t_lambda: type = 8; break;
      default:
         return;
   }

   switch( type )
   {
      case 0:
         m_out->writeString( " " + name + " " );
         gen_value( exp->first() );
      break;

      case 1:
         gen_value( exp->first() );
         m_out->writeString( " " + name + " " );
      break;

      case 2:
         m_out->writeString( "(" );
         gen_value( exp->first() );
         m_out->writeString( " " + name + " " );
         gen_value( exp->second() );
         m_out->writeString( " )" );
      break;

      case 3:
         m_out->writeString( "(" );
         gen_value( exp->first() );
         m_out->writeString( " ? (" );
         gen_value( exp->second() );
         m_out->writeString( " ):(" );
         gen_value( exp->third() );
         m_out->writeString( " ))" );
      break;

      case 4:
         m_out->writeString( "( " );
         gen_value( exp->first() );
         m_out->writeString( name );
         gen_value( exp->second() );
         m_out->writeString( " )" );
      break;

      case 5:
         gen_value( exp->first() );
         m_out->writeString( "[ " );
         gen_value( exp->second() );
         m_out->writeString( " ]" );
      break;

      case 6:
         gen_value( exp->first() );
         m_out->writeString( "." );
         gen_value( exp->second() );
      break;

      case 7:
         gen_value( exp->first() );
         m_out->writeString( "(" );
         if( exp->second() != 0 )
            gen_array( exp->second()->asArray() );
         m_out->writeString( " )" );
      break;

      case 8:
         m_out->writeString( exp->first()->asSymbol()->name() );
         if( exp->second() != 0 )
         {
            m_out->writeString( " closure (" );
            gen_array( exp->second()->asArray() );
            m_out->writeString( " )" );
         }

      break;

      case 10:
         gen_value( exp->first() );
         m_out->writeString( "[ *" );
         gen_value( exp->second() );
         m_out->writeString( " ]" );
      break;
   }
}

void GenTree::gen_array( const ArrayDecl *ad )
{
   ListElement *iter = ad->begin();
   while( iter != 0 )
   {
      const Value *val = (const Value *) iter->data();
      gen_value( val );
      iter = iter->next();
      if( iter != 0 )
         m_out->writeString( ", " );
   }
}

void GenTree::gen_dict( const DictDecl *ad )
{
   if( ad->empty() ) {
      m_out->writeString( " =>" );
      return;
   }

   ListElement *iter = ad->begin();
   while( iter != 0 )
   {
      DictDecl::pair *pair = (DictDecl::pair *) iter->data();
      const Value *key = pair->first;
      const Value *value = pair->second;

      gen_value( key );
      m_out->writeString( "=>" );
      gen_value( value );
      iter = iter->next();
      if( iter != 0 )
         m_out->writeString( ", " );
   }
}

}

/* end of gentree.cpp */
