/*
   FALCON - The Falcon Programming Language.
   FILE: gencode.cpp

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>
#include <falcon/pcodes.h>
#include <falcon/gencode.h>
#include <falcon/compiler.h>
#include <falcon/stream.h>
#include <falcon/fassert.h>
#include <falcon/linemap.h>

namespace Falcon
{

#define JMPTAG_UNDEFINED   0xFFFFFFFF


GenCode::c_jmptag::c_jmptag( Stream *stream, uint32 offset ):
   m_elifDefs( &traits::t_int() ),
   m_elifQueries( &traits::t_List() ),
   m_tries( 0 ),
   m_offset( offset ),
   m_bIsForLast( false ),
   m_ForinLoop(0),
   m_stream( stream )
{
   m_defs[0] = m_defs[1] = m_defs[2] = m_defs[3] = JMPTAG_UNDEFINED;
   m_elifQueries.resize(12);
}

uint32 GenCode::c_jmptag::addQuery( uint32 id, uint32 blocks )
{
   if ( m_defs[id] == JMPTAG_UNDEFINED )
   {
      uint32 pos = (uint32) m_stream->tell();
      m_queries[id].pushBack( pos + (blocks * 4) );
   }

   return m_defs[id];
}

uint32 GenCode::c_jmptag::addQueryElif( uint32 id, uint32 blocks )
{
   if ( m_elifDefs.size() <= id )
   {
      uint32 size = m_elifDefs.size();
      m_elifDefs.resize( id + 1 );

      uint32 undef = JMPTAG_UNDEFINED;
      while (size <= id )
         m_elifDefs.set( &undef, size ++ );

      if ( m_elifQueries.size() <= id )
         m_elifQueries.resize( id + 1 );
   }

   if ( *(uint32 *)m_elifDefs.at( id ) == JMPTAG_UNDEFINED )
   {
      uint32 pos = (uint32) m_stream->tell();
      ((List *)m_elifQueries.at(id))->pushBack( pos + (blocks * 4) );
   }

   return *(uint32 *) m_elifDefs.at(id);
}

void GenCode::c_jmptag::define( uint32 id )
{
   if ( m_defs[id] != JMPTAG_UNDEFINED )
      return;

   uint32 current_pos = (uint32) m_stream->tell();
   m_defs[id] = current_pos + m_offset;
   uint32 value = current_pos + m_offset;

   ListElement *iter = m_queries[id].begin();
   while( iter != 0 )
   {
      m_stream->seekBegin( iter->iData() );
      m_stream->write( &value, sizeof( value ) );
      iter = iter->next();
   }
   m_queries[id].clear();
   m_stream->seekBegin( current_pos );
}


void GenCode::c_jmptag::defineElif( uint32 id )
{
   if ( m_elifDefs.size() <= id )
   {
      uint32 size = m_elifDefs.size();
      m_elifDefs.resize( id + 1 );
      int32 undef = JMPTAG_UNDEFINED;
      while (size <= id )
         m_elifDefs.set( &undef, size ++ );

      if ( m_elifQueries.size() <= id )
         m_elifQueries.resize( id + 1 );
   }

   if ( *(uint32 *)m_elifDefs.at( id ) != JMPTAG_UNDEFINED )
      return;

   uint32 current_pos = (uint32) m_stream->tell();
   uint32 cp = current_pos + m_offset;
   m_elifDefs.set( &cp, id );
   uint32 value = current_pos + m_offset;
   List *lst = (List *) m_elifQueries.at(id);
   ListElement *iter = lst->begin();
   while( iter != 0 )
   {
      m_stream->seekBegin( iter->iData() );
      m_stream->write( reinterpret_cast< const char *>( &value ), sizeof( value ) );
      iter = iter->next();
   }
   lst->clear();
   m_stream->seekBegin( current_pos );
}



//===============================================================
// Code generator implementation
//===============================================================

GenCode::GenCode( Module *mod ):
   Generator( 0 ),
   m_pc(0),
   m_outTemp( new StringStream ),
   m_module( mod )
{}

GenCode::~GenCode()
{
   delete m_outTemp;
}

void GenCode::generate( const SourceTree *st  )
{
   // generates the main program
   if ( ! st->statements().empty() )
   {
      // entry point
      gen_block( &st->statements() );
      gen_pcode( P_RET );
      uint32 codeSize = m_outTemp->length();

      // create the main function.
      m_module->addFunction( "__main__", m_outTemp->closeToBuffer(), codeSize, false );
      m_pc += codeSize;
      delete m_outTemp;
      m_outTemp = new StringStream;
   }

   // No need to generate the classes, as they are just definitions in the
   // module symbols

   // generate functions
   if ( ! st->functions().empty() )
   {
      const StmtFunction *func = static_cast<const StmtFunction*>(st->functions().front());
      while( func != 0 )
      {
         gen_function( func );
         func = static_cast<const StmtFunction *>(func->next());
      }
   }
   else {
      // generate at least an valid VM stub
      if ( st->statements().empty() )
      {
         gen_pcode( P_RET );
      }
   }
}

void GenCode::gen_pcode( byte pcode,
            const c_varpar &first, const c_varpar &second, const c_varpar &third )
{
   byte pdefs[4];

   // first, ensure the third parameter is not complex.
   fassert( third.m_type != e_parVAL || third.m_content.value->isSimple() );

   c_varpar f1, s1;

   // then check for the need of a complex generation.
   if ( first.m_type == e_parVAL && ! first.m_content.value->isSimple() &&
      second.m_type == e_parVAL && ! second.m_content.value->isSimple() )
   {
      // generate first the first operand.
      gen_value( first.m_content.value );
      gen_pcode( P_PUSH, e_parA );
      // then generate the second operand.
      gen_value( second.m_content.value );
      gen_pcode( P_POP, e_parB );

      // and tell the rest of func to use B and A.
      f1 = e_parB;
      s1 = e_parA;
   }
   else if ( first.m_type == e_parVAL && ! first.m_content.value->isSimple() )
   {
      gen_value( first.m_content.value );
      f1 = e_parA;
      s1 = second;
   }
   else if ( second.m_type == e_parVAL && ! second.m_content.value->isSimple() )
   {
      gen_value( second.m_content.value );
      f1 = first;
      s1 = e_parA;
   }
   else {
      f1 = first;
      s1 = second;
   }

   // write the instruction header
   pdefs[0] = pcode;
   pdefs[1] = gen_pdef( f1 );
   pdefs[2] = gen_pdef( s1 );
   pdefs[3] = gen_pdef( third );
   m_outTemp->write( reinterpret_cast< const char * >( pdefs ), 4 );

   // now write the contents.
   f1.generate( this );
   s1.generate( this );
   third.generate( this );
}

void GenCode::c_varpar::generate( GenCode *owner ) const
{
   switch( m_type )
   {
      case e_parVAL:
         owner->gen_operand( m_content.value );
      break;

      case e_parVAR:
         owner->gen_var( *m_content.vd );
      break;

      case e_parSYM:
      {
         uint32 out = m_content.sym->itemId();
         owner->m_outTemp->write( &out, sizeof(out) );
      }
      break;

      case e_parSTRID:
      case e_parNTD32:
      case e_parINT32:
      {
         uint32 out = m_content.immediate;
         owner->m_outTemp->write( &out, sizeof(out) );
      }
      break;

      case e_parNTD64:
      {
         uint64 out = m_content.immediate64;
         owner->m_outTemp->write( &out, sizeof(out) );
      }
      break;

      default:
         break;
   }

   // else we should not generate anything
}


void GenCode::gen_operand( const Value *stmt )
{
   switch( stmt->type() )
   {
      case Value::t_symbol:
      {
         uint32 symid;
         const Symbol *sym = stmt->asSymbol();
         symid = sym->itemId();
         m_outTemp->write( &symid, sizeof( symid ) );
      }
      break;

      case Value::t_imm_integer:
      {
         int64 ival = stmt->asInteger();
         m_outTemp->write( &ival, sizeof( ival ) );
      }
      break;

      case Value::t_imm_num:
      {
         numeric dval = stmt->asNumeric();
         m_outTemp->write( &dval, sizeof( dval ) );
      }
      break;

      case Value::t_imm_string:
      {
         uint32 strid = (uint32) m_module->stringTable().findId( *stmt->asString() );
         m_outTemp->write( &strid, sizeof( strid ) );
      }
      break;

      case Value::t_lbind:
      {
         uint32 strid = (uint32) m_module->stringTable().findId( *stmt->asLBind() );
         m_outTemp->write( &strid, sizeof( strid ) );
      }
      break;

      case Value::t_imm_bool:
      case Value::t_self:
         // do nothing
      break;

      default:
         // can't be
         fassert(false);
   }
}

byte GenCode::gen_pdef( const c_varpar &elem )
{
   switch( elem.m_type )
   {
      case e_parND: return P_PARAM_NOTUSED;
      case e_parNIL: return P_PARAM_NIL;
      case e_parUND: return P_PARAM_UNB;
      case e_parVAL:
      {
         const Value *val = elem.m_content.value;
         switch( val->type() ) {
            case Value::t_nil: return P_PARAM_NIL;
            case Value::t_imm_bool: return val->asBool() ? P_PARAM_TRUE : P_PARAM_FALSE;
            case Value::t_imm_integer: return P_PARAM_INT64;
            case Value::t_imm_string: return P_PARAM_STRID;
            case Value::t_imm_num: return P_PARAM_NUM;
            case Value::t_self: return P_PARAM_REGS1;
            case Value::t_lbind: return P_PARAM_LBIND;

            case Value::t_symbol:
               if ( val->asSymbol()->isLocal() ) return P_PARAM_LOCID;
               if ( val->asSymbol()->isParam() ) return P_PARAM_PARID;
               // valid for undefined, extern and globals:
            return P_PARAM_GLOBID;

            default:
               break;
         }
      }
      // should not get here
      fassert( false );
      return P_PARAM_NOTUSED;

      case e_parVAR:
      {
         const VarDef *vd = elem.m_content.vd;
         switch( vd->type() ) {
            case VarDef::t_nil: return P_PARAM_NIL;
            case VarDef::t_bool: return vd->asBool() ? P_PARAM_TRUE : P_PARAM_FALSE;
            case VarDef::t_int: return P_PARAM_INT64;
            case VarDef::t_num: return P_PARAM_NUM;
            case VarDef::t_string: return P_PARAM_STRID;
            case Value::t_lbind: return P_PARAM_LBIND;
            case VarDef::t_symbol:
               if ( vd->asSymbol()->isLocal() ) return P_PARAM_LOCID;
               if ( vd->asSymbol()->isParam() ) return P_PARAM_PARID;
               // valid for undefined, extern and globals:
            return P_PARAM_GLOBID;

            default:
               break;
         }
      }
      // should not get here
      fassert( false );
      return P_PARAM_NOTUSED;

      case e_parSYM:
      {
         const Symbol *sym = elem.m_content.sym;
         if ( sym->isLocal() ) return P_PARAM_LOCID;
         if ( sym->isParam() ) return P_PARAM_PARID;
               // valid for undefined, extern and globals:
      }
      return P_PARAM_GLOBID;

      case e_parINT32: return P_PARAM_INT32;
      case e_parNTD32: return P_PARAM_NTD32;
      case e_parNTD64: return P_PARAM_NTD64;
      case e_parA: return P_PARAM_REGA;
      case e_parB: return P_PARAM_REGB;
      case e_parL1: return P_PARAM_REGL1;
      case e_parL2: return P_PARAM_REGL2;
      case e_parS1: return P_PARAM_REGS1;
      case e_parLBIND: return P_PARAM_LBIND;
      case e_parTRUE: return P_PARAM_TRUE;
      case e_parFALSE: return P_PARAM_FALSE;
      case e_parSTRID: return P_PARAM_STRID;
   }
   // should not get here
   fassert( false );
   return P_PARAM_NOTUSED;
}

void GenCode::gen_var( const VarDef &def )
{
   switch( def.type() )
   {
      case VarDef::t_int:
      {
         int64 ival = def.asInteger();
         m_outTemp->write( reinterpret_cast< const char *>( &ival ), sizeof( ival ) );
      }
      break;

      case VarDef::t_num:
      {
         numeric num = def.asNumeric();
         m_outTemp->write( reinterpret_cast< const char *>( &num ), sizeof( num ) );
      }
      break;

      case VarDef::t_string:
      {
         int32 id = m_module->stringTable().findId( *def.asString() );
         m_outTemp->write( reinterpret_cast< const char *>( &id ), sizeof( id ) );
      }
      break;
      case VarDef::t_reference:
      case VarDef::t_symbol:
      {
         const Symbol *sym = def.asSymbol();
         int32 ival = sym->itemId();
         m_outTemp->write( reinterpret_cast< const char *>( &ival ), sizeof( ival ) );
      }
      break;

      default:
         break;
   }
}



void GenCode::gen_function( const StmtFunction *func )
{
   m_functions.pushBack( func->symbol() );
   const StmtClass *ctorFor = func->constructorFor();
   byte ret_mode = ctorFor == 0 ? P_RET : P_RETV;

   // get the offset of the function
   const Symbol *funcsym = func->symbol();

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

   // picks a return mode, either ret/retv or state jump
   byte end_type = ret_mode;


   // Generates statements
   if ( ! func->staticBlock().empty() )
   {
      // push a label that is needed for the static block
      c_jmptag tag( m_outTemp ); // tag is destroed after pop
      m_labels.pushBack( &tag );

      gen_pcode( P_ONCE, c_param_fixed( tag.addQueryBegin() ), func->symbol() );
      gen_block( &func->staticBlock() );
      tag.defineBegin();

      // removes the static block tag
      m_labels.popBack();
   }

   if ( ! func->statements().empty() )
      gen_block( &func->statements() );

   if ( func->statements().empty() ||
      func->statements().back()->type() != Statement::t_return )
   {
      if ( end_type == P_RETV )
         gen_pcode( end_type, e_parS1 );
      else
         gen_pcode( end_type );
   }

   uint32 codeSize = m_outTemp->length();
   funcsym->getFuncDef()->codeSize( codeSize );
   funcsym->getFuncDef()->code( m_outTemp->closeToBuffer() );
   funcsym->getFuncDef()->basePC( m_pc );
   m_pc += codeSize;
   delete m_outTemp;
   m_outTemp = new StringStream;

   m_functions.popBack();
}

void GenCode::gen_block( const StatementList *slist )
{
   const Statement *stmt = slist->front();
   while( stmt != 0 ) {
      gen_statement( stmt );
      stmt = static_cast<const Statement *>(stmt->next());
   }
}

void GenCode::gen_statement( const Statement *stmt )
{
   static uint32 last_line=0;

   if ( stmt->line() != last_line )
   {
      last_line = stmt->line();
      m_module->addLineInfo( m_pc + (uint32) m_outTemp->tell(), last_line );
   }

   switch( stmt->type() )
   {
      case Statement::t_none:
         // ignore this statement.
      break;

      case Statement::t_break:
      case Statement::t_continue:
      {
         fassert( ! m_labels_loop.empty() );
         // first, eventually pop the needed tries.
         c_jmptag &tag = *(c_jmptag *) m_labels_loop.back();

         if ( tag.tryCount() > 0 )
            gen_pcode( P_PTRY, c_param_fixed( tag.tryCount() ) );

         // now jump to the loop begin or end.
         int32 jmppos;
         if ( stmt->type() == Statement::t_continue )
         {
            if( tag.isForIn() )
            {
               const StmtContinue *cont = static_cast<const StmtContinue *>( stmt );
               // last element has a special continue semantic.
               if( tag.isForLast() )
               {
                  // on dropping, just drop the last and go away
                  if( cont->dropping() )
                  {
                     gen_pcode( P_TRDN,
                       c_param_fixed( tag.addQueryEnd() ),
                       c_param_fixed( 0 ),
                       c_param_fixed( 0 ) );
                  }
                  else
                  {
                     // exactly as a break
                     jmppos = tag.addQueryEnd();
                     gen_pcode( P_JMP, c_param_fixed( jmppos ) );
                  }
               }
               else
               {
                  if( cont->dropping() )
                  {
                     const StmtForin* fi = tag.ForinLoop();
                     uint32 nv = fi->dest()->size();
                     gen_pcode( P_TRDN,
                          c_param_fixed( tag.addQueryBegin() ),
                          c_param_fixed( tag.addQueryEnd( 2 ) ),
                          c_param_fixed( nv ) );

                     ListElement* ite = fi->dest()->begin();
                     while( ite != 0 )
                     {
                        Value* v = (Value*) ite->data();
                        fassert( v->isSimple() );
                        gen_pcode( P_NOP, v );
                        ite = ite->next();
                     }
                  }
                  else {
                     // a normal continue -> next
                     jmppos = tag.addQueryNext();
                     gen_pcode( P_JMP, c_param_fixed( jmppos ) );
                  }
               }
            }
            else {
               jmppos = tag.addQueryNext();
               gen_pcode( P_JMP, c_param_fixed( jmppos ) );
            }
         }
         else {
            jmppos = tag.addQueryEnd();
            gen_pcode( P_JMP, c_param_fixed( jmppos ) );
         }
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
            gen_pcode( P_RET );
         }
         else {
            gen_pcode( P_RETV, ret->value() );
         }
      }
      break;

      case Statement::t_raise:
      {
         const StmtRaise *op = static_cast< const StmtRaise *>( stmt );
         gen_pcode( P_RIS, op->value() );
      }
      break;

      case Statement::t_fordot:
      {
         const StmtFordot *op = static_cast< const StmtFordot *>( stmt );
         gen_pcode( P_TRAC, op->value() );
      }
      break;

      case Statement::t_global:
      {
         // nothing to generate ??
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
            gen_pcode( P_WRT, val );
            iter = iter->next();
         }
      }
      break;

      case Statement::t_unref:
      {
         const StmtUnref *ass = static_cast<const StmtUnref *>(stmt);
         if( ass->symbol()->isSimple() ) {
            gen_pcode( P_LDRF, ass->symbol(), 0 );
         }
         else {
            gen_complex_value( ass->symbol() );
            gen_pcode( P_LDRF, e_parA, 0 );
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

         // create a new label
         c_jmptag tag( m_outTemp );
         m_labels.pushBack( &tag );

         gen_condition( elem->condition() );

         if( ! elem->children().empty() ) {
            gen_block( &elem->children() );
         }

         // do we need to jump away?
         if( ! (elem->elifChildren().empty() && elem->elseChildren().empty() ) ) {
            gen_pcode( P_JMP, c_param_fixed( tag.addQueryIfEnd() ) );
         }
         // this is the position for the failure of the condition.
         tag.defineIfElse();

         if ( ! elem->elifChildren().empty() )
         {
            const StmtElif *selif = static_cast<const StmtElif *>(elem->elifChildren().front());
            int elifcount = 0;

            while( selif != 0 )
            {
               gen_condition( selif->condition(), elifcount );
               if( ! selif->children().empty() )
               {
                  gen_block( &selif->children() );
                  // do we need to jump away?
                  if( selif->next() != 0 || ! elem->elseChildren().empty() )
                  {
                     gen_pcode( P_JMP, c_param_fixed( tag.addQueryIfEnd() ) );
                  }

               }
               // this is the end for this elif
               tag.defineElif( elifcount );
               elifcount++;
               selif = static_cast<const StmtElif *>(selif->next());
            }
         }

         if ( ! elem->elseChildren().empty() ) {
            gen_block( &elem->elseChildren() );
         }

         // this is the position for the end of the if.
         tag.defineIfEnd();

         m_labels.popBack();
      }
      break;

      case Statement::t_switch:
      case Statement::t_select:
      {
         const StmtSwitch *elem = static_cast<const StmtSwitch *>( stmt );

         // check for the item to be not empty. In that case, just
         // eventually generate tyhe switch expression.
         if ( elem->intCases().empty() && elem->rngCases().empty() &&
            elem->strCases().empty() && elem->objCases().empty() &&
            elem->defaultBlock().empty() &&
            elem->nilBlock() == -1 )
         {
            if ( ! elem->switchItem()->isSimple() )
               gen_complex_value( elem->switchItem() );
            break;
         }

         // first, build the switch descriptor
         int64 sizeInt = (int16) elem->intCases().size();
         int64 sizeRng = (int16) elem->rngCases().size();
         int64 sizeStr = (int16) elem->strCases().size();
         int64 sizeObj = (int16) elem->objCases().size();

         c_varpar sizes_par;
         sizes_par.m_type = e_parNTD64;
         sizes_par.m_content.immediate64 =
                  sizeInt << 48 | sizeRng << 32 | sizeStr << 16 | sizeObj;

         // prepare the jump out of the switch
         c_jmptag tag( m_outTemp );
         m_labels.pushBack( &tag );

         // gencode for the switch
         byte pCode = stmt->type() == Statement::t_switch ? P_SWCH : P_SELE;
         if( elem->switchItem()->isSimple() )
         {
            // the content doesn't matter. NTD32 will be fine.
            gen_pcode( pCode, c_param_fixed( tag.addQueryEnd() ), elem->switchItem(), sizes_par );
         }
         else {
            gen_value( elem->switchItem() );
            gen_pcode( pCode, c_param_fixed( tag.addQueryEnd() ), e_parA, sizes_par );
         }

         // prepare the nil block. May stay the dummy value if undefined.
         // and that is fine...
         int32 dummy = 0xFFFFFFFF;
         if ( elem->nilBlock() != -1 )
            tag.addQuerySwitchBlock( elem->nilBlock(),0 );
         m_outTemp->write( &dummy, sizeof( dummy ) );

         // write the int cases map
         MapIterator iter = elem->intCases().begin();
         while( iter.hasCurrent() )
         {
            Value *first = *(Value **) iter.currentKey();
            uint32 second = *(int32 *) iter.currentValue();
            int64 value = first->asInteger();
            m_outTemp->write( &value, sizeof(value ) );
            tag.addQuerySwitchBlock( second, 0 );
            m_outTemp->write( &dummy, sizeof( dummy ) );
            iter.next();
         }

         // write the range cases map
         iter = elem->rngCases().begin();
         while( iter.hasCurrent() )
         {
            Value *first = *(Value **) iter.currentKey();
            uint32 second = *(int32 *) iter.currentValue();
            int32 start = (uint32) first->asRange()->rangeStart()->asInteger();
            int32 end = (uint32) first->asRange()->rangeEnd()->asInteger();
            m_outTemp->write( &start, sizeof( start ) );
            m_outTemp->write( &end, sizeof( end ) );
            tag.addQuerySwitchBlock( second, 0 );
            m_outTemp->write( &dummy, sizeof( dummy ) );
            iter.next();
         }

         // write the string cases map
         iter = elem->strCases().begin();
         while( iter.hasCurrent() )
         {
            Value *first = *(Value **) iter.currentKey();
            uint32 second = *(int32 *) iter.currentValue();
            int32 strid = m_module->stringTable().findId( *first->asString() );
            m_outTemp->write( &strid, sizeof( strid ) );
            tag.addQuerySwitchBlock( second, 0 );
            m_outTemp->write( &dummy, sizeof( dummy ) );
            iter.next();
         }

         // we must put the objects in the same order they were declared.
         ListElement *obj_iter = elem->objList().begin();
         while( obj_iter != 0 )
         {
            Value *val = (Value *) obj_iter->data();
            if( elem->objCases().find( val, iter ) )
            {
               uint32 second = *(int32 *) iter.currentValue();
               Symbol *sym = val->asSymbol();
               int32 objid = sym->id();
               m_outTemp->write( &objid, sizeof( objid ) );
               tag.addQuerySwitchBlock( second, 0 );
               m_outTemp->write( &dummy, sizeof( dummy ) );
            }
            else {
               fassert( false );
            }
            obj_iter = obj_iter->next();
         }

         // generate the case blocks
         stmt = elem->blocks().front();
         int case_id = 0;
         while( stmt != 0 ) {
            const StmtCaseBlock *elem_case = static_cast<const StmtCaseBlock *>( stmt );
            // skip default block.
            tag.defineSwitchCase( case_id );
            gen_block( &elem_case->children() );
            // jump away, but only if needed.
            if ( elem_case->next() != 0 || ! elem->defaultBlock().empty() )
            {
               // ask for the postEnd in case of default block
               if ( elem->defaultBlock().empty() )
                  tag.addQueryEnd();
               else
                  tag.addQueryPostEnd();
               gen_pcode( P_JMP, e_parNTD32 );
            }
            case_id++;
            stmt = static_cast<const Statement *>(elem_case->next());
         }

         // end of the switch
         tag.defineEnd();
         // but eventually generate the default block
         if ( ! elem->defaultBlock().empty() ) {
            gen_block( &elem->defaultBlock() );
            tag.definePostEnd();
         }

         m_labels.popBack();
      }
      break;

      case Statement::t_while:
      {
         const StmtWhile *elem = static_cast<const StmtWhile *>( stmt );

         c_jmptag tag( m_outTemp );
         m_labels_loop.pushBack( &tag );

         // this is the place for next and begin
         tag.defineBegin();
         tag.defineNext();

         // generate condition check only if present and if not always true
         if ( elem->condition() != 0 && ! elem->condition()->isTrue() )
         {
            if ( elem->condition()->isSimple() ) {
               gen_pcode( P_IFF, c_param_fixed( tag.addQueryEnd() ), elem->condition() );
            }
            else {
               gen_complex_value( elem->condition() );
               gen_pcode( P_IFF, c_param_fixed( tag.addQueryEnd() ), e_parA );
            }
         }

         gen_block( &elem->children() );
         gen_pcode( P_JMP, c_param_fixed( tag.addQueryBegin() ) );
         // end of the loop
         tag.defineEnd();
         m_labels_loop.popBack();
      }
      break;

      case Statement::t_loop:
      {
         const StmtLoop *elem = static_cast<const StmtLoop *>( stmt );

         c_jmptag tag( m_outTemp );
         m_labels_loop.pushBack( &tag );

         // this is the place for begin
         tag.defineBegin();

         // if we don't have a check, continue can land at loop begin.
         if ( elem->condition() == 0 )
            tag.defineNext();

         gen_block( &elem->children() );

         // generate condition check only if present and if not always true
         if ( elem->condition() == 0 )
         {
            // endless loop
            gen_pcode( P_JMP, c_param_fixed( tag.addQueryBegin() ) );
         }
         else {
            tag.defineNext();

            if ( ! elem->condition()->isTrue() )
            {
               if ( elem->condition()->isSimple() ) {
                  gen_pcode( P_IFF, c_param_fixed( tag.addQueryBegin() ), elem->condition() );
               }
               else {
                  gen_complex_value( elem->condition() );
                  gen_pcode( P_IFF, c_param_fixed( tag.addQueryBegin() ), e_parA );
               }
            }
         }
         // if it's true, terminate immediately

         // end of the loop
         tag.defineEnd();
         m_labels_loop.popBack();
      }
      break;

      case Statement::t_propdef:
      {
         const StmtVarDef *pdef = static_cast<const StmtVarDef *>( stmt );
         if ( pdef->value()->isSimple() ) {
            gen_pcode( P_STP, e_parS1, c_param_str(
                  m_module->stringTable().findId( *pdef->name() )), pdef->value() );
         }
         else {
            gen_value( pdef->value() );
            gen_pcode( P_STP, e_parS1, c_param_str(
                  m_module->stringTable().findId( *pdef->name())), e_parA );
         }
      }
      break;

      case Statement::t_forin:
      {
         const StmtForin *loop = static_cast<const StmtForin *>( stmt );

         // begin a new loop
         c_jmptag tag( m_outTemp );
         tag.setForIn( loop );
         m_labels_loop.pushBack( &tag );

         uint32 neededVars = loop->dest()->size();

         if( loop->source()->isSimple() )
         {
            gen_pcode( P_TRAV,
                  c_param_fixed( tag.addQueryPostEnd() ),
                  c_param_fixed( neededVars ),
                  loop->source() );
         }
         else
         {
            gen_value( loop->source() );
            gen_pcode( P_TRAV,
                  c_param_fixed( tag.addQueryPostEnd() ),
                  c_param_fixed( neededVars ),
                  e_parA );
         }
            // generate all the NOPs
         ListElement* ite = loop->dest()->begin();
         while( ite != 0 )
         {
            Value* v = (Value*) ite->data();
            fassert( v->isSimple() );
            gen_pcode( P_NOP, v );
            ite = ite->next();
         }

         // have we got a "first" block?
         if ( ! loop->firstBlock().empty() ) {
            gen_block( &loop->firstBlock() );
         }

         // begin of the main loop;
         tag.defineBegin();
         if( ! loop->children().empty() ) {
            gen_block( &loop->children() );
         }

         // generate the formiddle block
         if( ! loop->middleBlock().empty() ) {
            // skip it for the last element
            gen_pcode( P_TRAL, c_param_fixed( tag.addQueryElif( 0 ) ) );

            gen_block( &loop->middleBlock() );
         }

         // after the formiddle block we have the next element pick
         tag.defineNext();

         gen_pcode( P_TRAN,
               c_param_fixed( tag.addQueryBegin() ),
               c_param_fixed( neededVars )
               );
         ite = loop->dest()->begin();
         while( ite != 0 )
         {
            Value* v = (Value*) ite->data();
            fassert( v->isSimple() );
            gen_pcode( P_NOP, v );
            ite = ite->next();
         }

         // and the last time...
         tag.defineElif( 0 );
         if( ! loop->lastBlock().empty() ) {
            tag.setForLast( true );
            gen_block( &loop->lastBlock() );
         }

         // create break landing code:
         tag.defineEnd();
         gen_pcode( P_IPOP, c_param_fixed(1) );
         tag.definePostEnd();

         m_labels_loop.popBack();
      }
      break;

      case Statement::t_try:
      {
         const StmtTry *op = static_cast< const StmtTry *>( stmt );
         // if the try block is empty we have nothing to do
         if( op->children().empty() )
            break;

         // also increment current loop try count if any.
         if ( ! m_labels_loop.empty() )
         ((c_jmptag *) m_labels_loop.back())->addTry();

         // we internally manage try landings, as they can be referenced only here.
         uint32 tryRefPos = ((uint32)m_outTemp->tell()) + 4;
         gen_pcode( P_TRY, c_param_fixed( 0xFFFFFFFF ) );

         // MUST maintain the current branch level to allow inner breaks.
         gen_block( &op->children() );

         // When the catcher is generated, the TRY cannot be broken anymore
         // by loop controls, as the catcher pops the TRY context from the VM
         if ( ! m_labels_loop.empty() )
            ((c_jmptag *) m_labels_loop.back())->removeTry();

         // start generating the blocks
         uint32 tryEndRefPos;
         if ( op->handlers().empty() && ! op->defaultGiven() ) {
            // we have no handlers
            gen_pcode( P_PTRY, c_param_fixed( 1 ) );
         }
         else {
            tryEndRefPos = ((uint32)m_outTemp->tell()) + 4;
            gen_pcode( P_JTRY, c_param_fixed( 0xFFFFFFFF ) );
         }

         // the landing is here
         uint32 curpos = (uint32) m_outTemp->tell();
         m_outTemp->seekBegin( tryRefPos );
         tryRefPos = curpos;
         m_outTemp->write( reinterpret_cast< const char *>(&tryRefPos), sizeof( tryRefPos ) );
         m_outTemp->seekBegin( curpos );

         // prepare the jump out of the switch, ok also if not used.
         c_jmptag tag( m_outTemp );
         m_labels.pushBack( &tag );

         if ( ! op->handlers().empty() || op->defaultGiven() )
         {
            // we have some handlers. If we have only the default, we don't create a select.
            if ( ! op->handlers().empty() )
            {
               // we have to generate a SELE on m_regB
               // first, build the switch descriptor
               int64 sizeInt = (int16) op->intCases().size();
               int64 sizeObj = (int16) op->objCases().size();

               c_varpar sizes_par;
               sizes_par.m_type = e_parNTD64;
               sizes_par.m_content.immediate64 = sizeInt << 48 | sizeObj;


               // gencode for the switch
               gen_pcode( P_SELE, c_param_fixed( tag.addQueryEnd() ), e_parB, sizes_par );

               // prepare the nil block -- used in select B to determine if we should re-raise
               int32 dummy = op->defaultGiven() ? 1:0;
               m_outTemp->write( &dummy, sizeof( dummy ) );

               // write the int cases map
               MapIterator iter = op->intCases().begin();
               while( iter.hasCurrent() )
               {
                  Value *first = *(Value **) iter.currentKey();
                  uint32 second = *(int32 *) iter.currentValue();
                  int64 value = first->asInteger();
                  m_outTemp->write( &value, sizeof(value ) );
                  tag.addQuerySwitchBlock( second, 0 );
                  m_outTemp->write( &dummy, sizeof( dummy ) );
                  iter.next();
               }

               // we must put the objects in the same order they were declared.
               ListElement *obj_iter = op->objList().begin();
               while( obj_iter != 0 )
               {
                  Value *val = (Value *) obj_iter->data();
                  if( op->objCases().find( val, iter ) )
                  {
                     uint32 second = *(int32 *) iter.currentValue();
                     Symbol *sym = val->asSymbol();
                     int32 objid = sym->id();
                     m_outTemp->write( &objid, sizeof( objid ) );
                     tag.addQuerySwitchBlock( second, 0 );
                     m_outTemp->write( &dummy, sizeof( dummy ) );
                  }
                  else {
                     fassert( false );
                  }
                  obj_iter = obj_iter->next();
               }

               // generate the case blocks
               stmt = op->handlers().front();
               int case_id = 0;
               while( stmt != 0 ) {
                  const StmtCatchBlock *elem_case = static_cast<const StmtCatchBlock *>( stmt );
                  // skip default block.
                  tag.defineSwitchCase( case_id );
                  // if we have a catch into, we have to move B here.
                  if ( elem_case->intoValue() != 0 )
                     gen_pcode( P_LD, elem_case->intoValue(), e_parB );

                  gen_block( &elem_case->children() );
                  // jump away, but only if needed.
                  if ( elem_case->next() != 0 || op->defaultGiven() )
                  {
                     // ask for the postEnd in case of default block
                     if ( ! op->defaultGiven() )
                        tag.addQueryEnd();
                     else
                        tag.addQueryPostEnd();
                     gen_pcode( P_JMP, e_parNTD32 );
                  }
                  case_id++;
                  stmt = static_cast<const Statement *>(elem_case->next());
               }

            }

            // end of the switch
            tag.defineEnd();   // tell the news to the other branches

            // but eventually generate the default block
            if ( op->defaultGiven() ) {
               if ( op->defaultHandler()->intoValue() != 0 )
                  gen_pcode( P_LD, op->defaultHandler()->intoValue(), e_parB );
               gen_block( &op->defaultHandler()->children() );

               // even if we don't use it, no one will care.
               tag.definePostEnd();
            }

            m_labels.popBack();

            // tell the TRY jumper where to jump when we are out of TRY
            curpos = (uint32) m_outTemp->tell();
            m_outTemp->seekBegin( tryEndRefPos );
            tryEndRefPos = curpos;
            m_outTemp->write( reinterpret_cast< const char *>(&tryEndRefPos), sizeof( tryEndRefPos ) );
            m_outTemp->seekBegin( curpos );
         }
      }
      break;

      default:
         break;
   }
}


void GenCode::gen_inc_prefix( const Value *val )
{
   if ( val->isSimple() ) {
      gen_pcode( P_INC, val );
   }
   else {
      t_valType xValue = l_value;
      gen_complex_value( val, xValue );
      gen_pcode( P_INC, e_parA );

      if ( xValue == p_value )
         gen_pcode( P_STP, e_parL1, e_parL2, e_parA );
      else if ( xValue == v_value )
         gen_pcode( P_STV, e_parL1, e_parL2, e_parA );
   }
}

void GenCode::gen_inc_postfix( const Value *val )
{
   if ( val->isSimple() ) {
      gen_pcode( P_INCP, val );
   }
   else {
      t_valType xValue = l_value;
      gen_complex_value( val, xValue );
      gen_pcode( P_INCP, e_parA );

      if ( xValue == p_value )
         gen_pcode( P_STP, e_parL1, e_parL2, e_parB );
      else if ( xValue == v_value )
         gen_pcode( P_STV, e_parL1, e_parL2, e_parB );
   }
}

void GenCode::gen_dec_prefix( const Value *val )
{
   if ( val->isSimple() ) {
      gen_pcode( P_DEC, val );
   }
   else {
      t_valType xValue = l_value;
      gen_complex_value( val, xValue );
      gen_pcode( P_DEC, e_parA );

      if ( xValue == p_value )
         gen_pcode( P_STP, e_parL1, e_parL2, e_parA );
      else if ( xValue == v_value )
         gen_pcode( P_STV, e_parL1, e_parL2, e_parA );
   }
}

void GenCode::gen_dec_postfix( const Value *val )
{
   if ( val->isSimple() ) {
      gen_pcode( P_DECP, val );
   }
   else {
      t_valType xValue = l_value;
      gen_complex_value( val, xValue );
      gen_pcode( P_DECP, e_parA );

      if ( xValue == p_value )
         gen_pcode( P_STP, e_parL1, e_parL2, e_parB );
      else if ( xValue == v_value )
         gen_pcode( P_STV, e_parL1, e_parL2, e_parB );
   }
}

void GenCode::gen_autoassign( byte opcode, const Value *target, const Value *source )
{
   if( target->isSimple() && source->isSimple() ) {
      gen_pcode( opcode, target, source );
   }
   else if ( target->isSimple() )
   {
      gen_complex_value( source );
      gen_pcode( opcode, target, e_parA );
   }
   else if ( source->isSimple() )
   {
      t_valType xValue = l_value;
      gen_complex_value( target, xValue );

      gen_pcode( opcode, e_parA, source );

      if ( xValue == p_value )
         gen_pcode( P_STP, e_parL1, e_parL2, e_parA );
      else if ( xValue == v_value )
         gen_pcode( P_STV, e_parL1, e_parL2, e_parA );
   }
   else {
      gen_complex_value( source );
      gen_pcode( P_PUSH, e_parA );

      t_valType xValue = l_value;
      gen_complex_value( target, xValue );

      gen_pcode( P_POP, e_parB );
      gen_pcode( opcode, e_parA, e_parB );

      if ( xValue == p_value )
         gen_pcode( P_STP, e_parL1, e_parL2, e_parA );
      else if ( xValue == v_value )
         gen_pcode( P_STV, e_parL1, e_parL2, e_parA );
   }
}

void GenCode::gen_condition( const Value *stmt, int mode )
{
   c_jmptag &tag = *(c_jmptag*) m_labels.back();
   if ( !stmt->isSimple() )
   {
      gen_complex_value( stmt );
      if ( mode >= 0 ) {
         gen_pcode( P_IFF, c_param_fixed( tag.addQueryElif( mode ) ), e_parA );
      }
      else {
         gen_pcode( P_IFF, c_param_fixed( tag.addQueryIfElse() ), e_parA );
      }
   }
   else
   {
      if ( mode >= 0 ) {
         gen_pcode( P_IFF, c_param_fixed( tag.addQueryElif( mode ) ), stmt );
      }
      else {
         gen_pcode( P_IFF, c_param_fixed( tag.addQueryIfElse() ), stmt );
      }
   }
}

void GenCode::gen_value( const Value *stmt )
{
   if ( stmt->isSimple() ) {
      gen_pcode( P_STO, e_parA, stmt );
   }
   else {
      gen_complex_value( stmt );
   }
}



void GenCode::gen_complex_value( const Value *stmt, t_valType &xValue )
{
   switch( stmt->type() )
   {
      // catch also reference taking in case it's not filtered before
      // this happens when we have not a specific opcode to handle references
      case Value::t_byref:
         if ( stmt->asReference()->isSymbol() )
            gen_pcode( P_LDRF, e_parA, stmt->asReference()->asSymbol() );
         else {
            gen_value( stmt->asReference() );
            // won't do a lot, but we need it
            gen_pcode( P_LDRF, e_parA, e_parA );
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
         fassert( false );
   }
}


void GenCode::gen_push( const Value *val )
{
   if( val->isSimple() ) {
      gen_pcode( P_PUSH, val );
   }
   else if ( val->isReference() ) {
      if( val->asReference()->isSymbol() )
         gen_pcode( P_PSHR, val->asReference()->asSymbol() );
      else {
         gen_value( val->asReference() );
         gen_pcode( P_PSHR, e_parA );
      }

   }
   else {
      gen_complex_value( val );
      gen_pcode( P_PUSH, e_parA );
   }
}


void GenCode::gen_expression( const Expression *exp, t_valType &xValue )
{

   byte opname = 0;
   int mode = 0; // 1 = unary, 2 = binary

   // first, deterime the operator name and operation type
   switch( exp->type() )
   {
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
         char ifmode;

         if ( exp->type() == Expression::t_or )
         {
            opname = P_OR;
            ifmode = P_IFT;
         }
         else {
            opname = P_AND;
            ifmode = P_IFF;
         }

         if( exp->first()->isSimple() && exp->second()->isSimple() )
         {
            gen_pcode( opname, exp->first(), exp->second() );
         }
         else
         {
            c_jmptag tag( m_outTemp );
            m_labels.pushBack( &tag );

            if( exp->first()->isSimple() )
            {
               gen_pcode( P_LD, e_parA, exp->first() );
               gen_pcode( ifmode, c_param_fixed( tag.addQueryIfEnd() ), e_parA );
               // if A that is the first op is false, we generate this one.
               gen_value( exp->second() );
            }
            else if( exp->second()->isSimple() )
            {
               gen_value( exp->first() );
               gen_pcode( ifmode, c_param_fixed( tag.addQueryIfEnd() ), e_parA );
               // if A that is the first op is false, we generate this one.
               gen_pcode( P_LD, e_parA, exp->second() );
            }
            else
            {
               gen_value( exp->first() );
               gen_pcode( ifmode, c_param_fixed( tag.addQueryIfEnd() ), e_parA );
               // if A that is the first op is false, we generate this one.
               gen_value( exp->second() );
            }
            tag.defineIfEnd();
            m_labels.popBack();
         }
      }
      return;

      // unary operators
      case Expression::t_neg: mode = 1; opname = P_NEG; break;
      case Expression::t_not: mode = 1; opname = P_NOT; break;
      case Expression::t_bin_not: mode = 1; opname = P_BNOT; break;
      case Expression::t_strexpand: mode = 1; opname = P_STEX; break;
      case Expression::t_indirect: mode = 1; opname = P_INDI; break;
      case Expression::t_eval: mode = 1; opname = P_EVAL; break;

      case Expression::t_bin_and: mode = 2; opname = P_BAND; break;
      case Expression::t_bin_or: mode = 2; opname = P_BOR; break;
      case Expression::t_bin_xor: mode = 2; opname = P_BXOR; break;
      case Expression::t_shift_left: mode = 2; opname = P_SHL; break;
      case Expression::t_shift_right: mode = 2; opname = P_SHR; break;

      case Expression::t_plus: mode = 2; opname = P_ADD; break;
      case Expression::t_minus: mode = 2; opname = P_SUB; break;
      case Expression::t_times: mode = 2; opname = P_MUL; break;
      case Expression::t_divide: mode = 2;opname = P_DIV; break;
      case Expression::t_modulo: mode = 2; opname = P_MOD; break;
      case Expression::t_power: mode = 2; opname = P_POW; break;

      case Expression::t_gt: mode = 2; opname = P_GT; break;
      case Expression::t_ge: mode = 2; opname = P_GE; break;
      case Expression::t_lt: mode = 2; opname = P_LT; break;
      case Expression::t_le: mode = 2; opname = P_LE; break;
      case Expression::t_eq: mode = 2; opname = P_EQ; break;
      case Expression::t_exeq: mode = 2; opname = P_EXEQ; break;
      case Expression::t_neq: mode = 2; opname = P_NEQ; break;

      case Expression::t_in: mode = 2; opname = P_IN; break;
      case Expression::t_notin: mode = 2; opname = P_NOIN; break;
      case Expression::t_provides: mode = 2; opname = P_PROV; break;
      case Expression::t_fbind: mode = 2; opname = P_FORB; break;

      // it is better to handle the rest directly here.
      case Expression::t_iif:
      {
         xValue = l_value;
         c_jmptag tag( m_outTemp );
         m_labels.pushBack( &tag );

         // condition
         if ( exp->first()->isSimple() )
         {
            tag.addQueryIfElse();
            gen_pcode( P_IFF, e_parNTD32, exp->first() );
         }
         else {
            gen_value( exp->first() );
            tag.addQueryIfElse();
            gen_pcode( P_IFF, e_parNTD32, e_parA );
         }

         // success case
         if( exp->second()->isSimple() )
         {
            gen_pcode( P_STO, e_parA, exp->second() );
         }
         else
         {
            gen_value( exp->second() );
         }

         tag.addQueryIfEnd();
         gen_pcode( P_JMP, e_parNTD32 );
         tag.defineIfElse();

         //failure case
         // success case
         if( exp->third()->isSimple() )
         {
            gen_pcode( P_STO, e_parA, exp->third() );
         }
         else
         {
            gen_value( exp->third() );
         }

         tag.defineIfEnd();
         m_labels.popBack();
      }
      return;

      case Expression::t_assign:
         // handle it as a load...
         // don't change x-value
         gen_load( exp->first(), exp->second() );
      return;

      case Expression::t_aadd:
         xValue = l_value;
         gen_autoassign( P_ADDS, exp->first(), exp->second() );
      return;

      case Expression::t_asub:
         xValue = l_value;
         gen_autoassign( P_SUBS, exp->first(), exp->second() );
      return;

      case Expression::t_amul:
         xValue = l_value;
         gen_autoassign( P_MULS, exp->first(), exp->second() );
      return;

      case Expression::t_adiv:
         xValue = l_value;
         gen_autoassign( P_DIVS, exp->first(), exp->second() );
      return;

      case Expression::t_amod:
         xValue = l_value;
         gen_autoassign( P_MODS, exp->first(), exp->second() );
      return;

      case Expression::t_aband:
         xValue = l_value;
         gen_autoassign( P_ANDS, exp->first(), exp->second() );
      return;

      case Expression::t_abor:
         xValue = l_value;
         gen_autoassign( P_ORS, exp->first(), exp->second() );
      return;

      case Expression::t_abxor:
         xValue = l_value;
         gen_autoassign( P_XORS, exp->first(), exp->second() );
      return;

      case Expression::t_apow:
         xValue = l_value;
         gen_autoassign( P_POWS, exp->first(), exp->second() );
      return;

      case Expression::t_ashl:
         xValue = l_value;
         gen_autoassign( P_SHLS, exp->first(), exp->second() );
      return;

      case Expression::t_ashr:
         xValue = l_value;
         gen_autoassign( P_SHRS, exp->first(), exp->second() );
      return;

      case Expression::t_pre_inc: gen_inc_prefix( exp->first() ); return;
      case Expression::t_pre_dec: gen_dec_prefix( exp->first() ); return;
      case Expression::t_post_inc: gen_inc_postfix( exp->first() ); return;
      case Expression::t_post_dec: gen_dec_postfix( exp->first() ); return;

      case Expression::t_obj_access:
         xValue = p_value;
         gen_load_from_deep( P_LDP, exp->first(), exp->second() );
      return;

      case Expression::t_array_access:
         xValue = v_value;
         gen_load_from_deep( P_LDV, exp->first(), exp->second() );
      return;

      case Expression::t_array_byte_access:
         xValue = l_value;
         gen_load_from_deep( P_LSB, exp->first(), exp->second() );
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
         xValue = l_value;
         if( exp->second() != 0 )
         {
            // we must create the lambda closure
            int size = 0;
            ListElement *iter = exp->second()->asArray()->begin();
            while( iter != 0 )
            {
               const Value *val = (Value *) iter->data();
               // push the reference; we want a reference in the closure.
               gen_pcode( P_PSHR, val );
               size++;
               iter = iter->next();
            }

            gen_pcode( P_CLOS, c_param_fixed(size), e_parA, exp->first()->asSymbol() );
         }
         else
         {
            gen_pcode( P_STO, e_parA, exp->first()->asSymbol() );
         }
      return;

      case Expression::t_oob:
         xValue = l_value;
         gen_pcode( P_OOB, c_param_fixed(1), exp->first() );
      return;

      case Expression::t_deoob:
         xValue = l_value;
         gen_pcode( P_OOB, c_param_fixed(0), exp->first() );
      return;

      case Expression::t_xoroob:
         xValue = l_value;
         gen_pcode( P_OOB, c_param_fixed(2), exp->first() );
      return;

      case Expression::t_isoob:
         xValue = l_value;
         gen_pcode( P_OOB, c_param_fixed(3), exp->first() );
      return;

      default:
         break;
   }

   // post-processing unary and binary operators.
   // ++, -- and accessors are gone; safely change the l-value status.
   xValue = l_value;

   // then, if there is still something to do, put the operands in place.
   if ( mode == 1 ) {  // unary?
      if ( exp->first()->isSimple() ) {
         gen_pcode( opname, exp->first() );
      }
      else {
         gen_complex_value( exp->first() );
         gen_pcode( opname, e_parA );
      }
   }
   else {
      if ( exp->first()->isSimple() && exp->second()->isSimple() ) {
         gen_pcode( opname, exp->first(), exp->second() );
      }
      else if ( exp->first()->isSimple() ) {
         gen_complex_value( exp->second() );
         gen_pcode( opname, exp->first(), e_parA );
      }
      else if ( exp->second()->isSimple() ) {
         gen_complex_value( exp->first() );
         gen_pcode( opname, e_parA, exp->second() );
      }
      else {
         gen_complex_value( exp->first() );
         gen_pcode( P_PUSH, e_parA );
         gen_complex_value( exp->second() );
         gen_pcode( P_POP, e_parB );
         gen_pcode( opname, e_parB, e_parA );
      }
   }

}

void GenCode::gen_dict_decl( const DictDecl *dcl )
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

   gen_pcode( P_GEND, c_param_fixed( size ) );
}

void GenCode::gen_array_decl( const ArrayDecl *dcl )
{
   int size = 0;

   ListElement *iter = dcl->begin();
   while( iter != 0 )
   {
      const Value *val = (Value *) iter->data();
      gen_push( val );
      size++;
      iter = iter->next();
   }
   gen_pcode( P_GENA, c_param_fixed( size ) );
}


void GenCode::gen_range_decl( const RangeDecl *dcl )
{
   if ( dcl->isOpen() )
   {
      if ( dcl->rangeStart()->isSimple() ) {
         gen_pcode( P_GEOR, dcl->rangeStart() );
      }
      else {
         gen_complex_value( dcl->rangeStart() );
         gen_pcode( P_GEOR, e_parA );
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
         gen_pcode( P_PUSH, e_parA );
         // we'll instruct GENR to get it via NIL as parameter
         rangeStep = &dummy;
      }

      if ( dcl->rangeStart()->isSimple() && dcl->rangeEnd()->isSimple() )
      {
         gen_pcode( P_GENR, dcl->rangeStart(), dcl->rangeEnd(), rangeStep );
      }
      else if ( dcl->rangeStart()->isSimple() )
      {
         gen_complex_value( dcl->rangeEnd() );
         gen_pcode( P_GENR, dcl->rangeStart(), e_parA, rangeStep );
      }
      else if ( dcl->rangeEnd()->isSimple() )
      {
         gen_complex_value( dcl->rangeStart() );
         gen_pcode( P_GENR, e_parA, dcl->rangeEnd(), rangeStep );

      }
      else {
         gen_complex_value( dcl->rangeStart() );
         gen_pcode( P_PUSH, e_parA );
         gen_complex_value( dcl->rangeEnd() );
         gen_pcode( P_POP, e_parB );
         gen_pcode( P_GENR, e_parB, e_parA, rangeStep );
      }
   }
}

void GenCode::gen_load( const Value *target, const Value *source )
{
   if ( target->isSimple() && source->isSimple() )
   {
      gen_pcode( P_LD, target, source );
   }
   else if ( target->isSimple() )
   {
      if( source->isReference() )
      {
         if ( source->asReference() == 0 )
            gen_pcode( P_LDRF, target, 0 );
         else {
            if( source->asReference()->isSymbol() )
            {
               gen_pcode( P_LDRF, target, source->asReference()->asSymbol() );
            }
            else {
               gen_value( source->asReference() );
               gen_pcode( P_LDRF, target, e_parA );
            }
         }
      }
      else {
         gen_complex_value( source );
         gen_pcode( P_LD, target, e_parA );
      }
   }
   else {
      // target is NOT simple. If it's an expression ...
      if( target->type() == Value::t_expression )
      {
         const Expression *exp = target->asExpr();
         // ... then it may be an array assignment...

         if( exp->type() == Expression::t_array_access ) {
            gen_store_to_deep( P_STV, exp->first(), exp->second(), source );
         }
         else if ( exp->type() == Expression::t_obj_access ) {
            gen_store_to_deep( P_STP, exp->first(), exp->second(), source );
         }
      }
      else if ( target->type() == Value::t_array_decl )
      {
         const ArrayDecl *tarr = target->asArray();

         // if the source is also an array, fine, we have a 1:1 assignment.
         if ( source->type() == Value::t_array_decl )
         {
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
         else {
            // then unpack the source in the array.
            if ( source->isSimple() ) {
               gen_pcode( P_LDAS, c_param_fixed( tarr->size() ), source );
            }
            else {
					gen_complex_value( source );
					gen_pcode( P_LDAS, c_param_fixed( tarr->size() ), e_parA );
            }

            ListElement *it_t = tarr->end();
            while( it_t != 0 ) {
               const Value *t = (const Value *) it_t->data();
               if( t->isSimple() )
                   gen_pcode( P_POP, t );
               else {
                  gen_pcode( P_POP, e_parB );
                  gen_load_from_reg( t, e_parB );
               }
               it_t = it_t->prev();
            }
         }
      }
   }
}

int GenCode::gen_refArray( const ArrayDecl *tarr, bool bGenArray )
{
   ListElement *it_t = tarr->begin();
   int size = 0;

   // first generates an array of references
   while( it_t != 0 ) {
      // again, is the compiler that must make sure of this...
      const Value *val = (const Value *) it_t->data();
      fassert( val->isSimple() );

      gen_pcode( P_PSHR, val );
      ++size;
      it_t = it_t->next();
   }

	if( bGenArray )
		gen_pcode( P_GENA, c_param_fixed( size ) );
   return size;
}

void GenCode::gen_store_to_deep( byte type, const Value *first, const Value *second, const Value *source )
{
   // first we must generate the assignands.
   if( source->isSimple() )
   {
      if ( first->isSimple() && second->isSimple() ) {
         gen_pcode( type, first, second, source );
      }
      else if ( second->isSimple() ) {
         gen_complex_value( first );
         gen_pcode( type, e_parA, second, source );
      }
      else if ( first->isSimple() ) {
         gen_complex_value( second );
         gen_pcode( type, first, e_parA, source );
      }
      else {
         gen_complex_value( first );
         gen_pcode( P_PUSH, e_parA );
         gen_complex_value( second );
         gen_pcode( P_POP, e_parB );
         gen_pcode( type, e_parB, e_parA, source );
      }
   }
   else {
      if ( first->isSimple() && second->isSimple() )
      {
         gen_complex_value( source );
         gen_pcode( type, first, second, e_parA );
      }
      else {
         if( source->isReference() ) {
            type == P_STP ? type = P_STPR : P_STVR;
            source = source->asReference();
         }
         else {
            type = type == P_STP ? P_STPS : P_STVS;
         }

         if ( second->isSimple() ) {
            gen_complex_value( first );
            gen_pcode( P_PUSH, e_parA );
            gen_complex_value( source );
            gen_pcode( P_XPOP, e_parA );
            gen_pcode( type, e_parA, second );
         }
         else if ( first->isSimple() ) {
            gen_complex_value( second );
            gen_pcode( P_PUSH, e_parA );
            gen_complex_value( source );
            gen_pcode( P_XPOP, e_parA );
            gen_pcode( type, first, e_parA );
         }
         else {
            gen_complex_value( first );
            gen_pcode( P_PUSH, e_parA );
            gen_complex_value( second );
            gen_pcode( P_PUSH, e_parA );
            gen_complex_value( source );
            gen_pcode( P_POP, e_parB );
            gen_pcode( P_XPOP, e_parA );
            gen_pcode( type, e_parA, e_parB );
         }
      }
   }
}


void GenCode::gen_load_from_deep( byte type, const Value *first, const Value *second )
{

   // first we must generate the assignands.
   if( first->isSimple() )
   {
      if ( second->isSimple() ) {
         gen_pcode( type, first, second );
      }
      else {
         gen_complex_value( second );
         gen_pcode( type, first, e_parA );
      }
   }
   else {
      if ( second->isSimple() )
      {
         gen_complex_value( first );
         gen_pcode( type, e_parA, second );
      }
      else {
         gen_complex_value( first );
         gen_pcode( P_PUSH, e_parA );
         gen_complex_value( second );
         gen_pcode( P_POP, e_parB );
         gen_pcode( type, e_parB, e_parA );
      }
   }
}


void GenCode::gen_load_from_A( const Value *target )
{
   gen_load_from_reg( target, e_parA );
}

void GenCode::gen_load_from_reg( const Value *target, t_paramType reg )
{
   if ( target->isSimple() )
   {
      gen_pcode( P_LD, target, reg );
   }
   else {
      // target is NOT simple. If it's an expression ...
      if( target->type() == Value::t_expression )
      {
         const Expression *exp = target->asExpr();
         // ... then it may be an array assignment...

         if( exp->type() == Expression::t_array_access ) {
            gen_store_to_deep_reg( P_STV, exp->first(), exp->second(), reg );
         }
         else if ( exp->type() == Expression::t_obj_access ) {
            gen_store_to_deep_reg( P_STP, exp->first(), exp->second(), reg );
         }
         else {
            gen_pcode( P_PUSH, reg );
            gen_expression( exp );
            gen_pcode( P_POP, e_parB );
            gen_pcode( P_LD, e_parA, e_parB );
         }
      }
      else if ( target->type() == Value::t_array_decl )
      {
         // if the source is also an array, fine, we have a 1:1 assignment.
         const ArrayDecl *tarr = target->asArray();
         // push the source array on the stack.
         gen_pcode( P_LDAS, c_param_fixed( tarr->size() ), reg );

         // and load each element back in the array
         ListElement *it_t = tarr->end();

         // Now load each element by popping it.
         while( it_t != 0 )
         {
            const Value *val = (const Value *) it_t->data();
            if( val->isSimple() )
            {
               gen_pcode( P_POP, val );
            }
            else {
               gen_pcode( P_POP, e_parB );
               gen_load_from_reg( val, e_parB );
            }
            it_t = it_t->prev();
         }
      }
   }
}

void GenCode::gen_store_to_deep_A( byte type, const Value *first, const Value *second )
{
   gen_store_to_deep_reg( type, first, second, e_parA );
}

void GenCode::gen_store_to_deep_reg( byte type, const Value *first, const Value *second, t_paramType reg )
{

   // first we must generate the assignands.
   if ( first->isSimple() && second->isSimple() ) {
      gen_pcode( type, first, second, reg );
   }
   else
   {
      type = type == P_STP ? P_STPS : P_STVS;
      gen_pcode( P_PUSH, reg );

      if ( second->isSimple() ) {
         gen_complex_value( first );
         gen_pcode( type, e_parA, second );
      }
      else if ( first->isSimple() )
      {
         gen_complex_value( second );
         gen_pcode( type, first, reg );
      }
      else {
         gen_complex_value( first );
         gen_pcode( P_PUSH, e_parA );
         gen_complex_value( second );
         gen_pcode( P_POP, e_parB );
         gen_pcode( type, e_parB, e_parA );
      }
   }
}


void GenCode::gen_funcall( const Expression *exp, bool fork )
{
   int size = 0;

   byte functor = exp->type() == Expression::t_inherit ? P_INST : P_CALL;

   if( exp->second() != 0 )
   {
      const ArrayDecl *dcl = exp->second()->asArray();
      ListElement *iter = (ListElement *) dcl->begin();
      while( iter != 0 )
      {
         const Value *val = (const Value *) iter->data();
         gen_push( val );
         size++;
         iter = iter->next();
      }
   }

   uint32 fork_pos;
   if ( fork ) {
      // push the function call
      if ( exp->first()->isSimple() ) {
         gen_pcode( P_PUSH, exp->first() );
      }
      else {
         gen_complex_value( exp->first() );
         gen_pcode( P_PUSH, e_parA );
      }
      fork_pos = ((uint32)m_outTemp->tell()) + 8;
      gen_pcode( P_FORK, c_param_fixed( size + 1 ), e_parNTD32 );
      // call the topmost stack item.
      gen_pcode( P_POP, e_parA );
      gen_pcode( functor, c_param_fixed( size ), e_parA );

      gen_pcode( P_END );
      // landing for fork
      uint32 curpos = (uint32) m_outTemp->tell();
      m_outTemp->seekBegin( fork_pos );
      fork_pos = curpos;
      m_outTemp->write( &fork_pos, sizeof( fork_pos ) );
      m_outTemp->seekBegin( curpos );
   }
   else {
      if( exp->first()->isSimple() ) {
         gen_pcode( functor, c_param_fixed( size ), exp->first() );
      }
      else {
         gen_complex_value( exp->first() );
         gen_pcode( functor, c_param_fixed( size ), e_parA );
      }
   }
}

}

/* end of gencode.cpp */
