/*
   FALCON - The Falcon Programming Language.
   FILE: faldisass.cpp

   Falcon disassembler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2004-08-01

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>
#include <falcon/fstream.h>
#include <falcon/stdstreams.h>
#include <falcon/stringstream.h>
#include <falcon/transcoding.h>

#include <vector>
#include <list>
#include <map>

#include "faldisass.h"

namespace Falcon
{

void gen_code( Module *mod, const FuncDef *fd, Stream *out, const t_labelMap &labels );

SymbolTable *global_symtab;
SymbolTable *current_symtab;

/**************************************************************
* Option management
***************************************************************/
Options::Options():
   m_dump_str( false ),
   m_dump_sym( false ),
   m_dump_dep( false ),
   m_isomorphic( false ),
   m_inline_str( false ),
   m_stdin( false ),
   m_lineinfo( true ),
   m_fname(0)
{}

Options options;


/**************************************************************
*
***************************************************************/

void write_label( Stream *output, uint32 landing )
{
   String temp;
   if ( options.m_isomorphic )
   {
      temp = "_label_";
      temp.writeNumber( (int64) landing );
   }
   else {
      temp = "0x";
      temp.writeNumberHex( landing, true );
   }

   output->writeString( temp );
}

void write_label_long( Stream *output, uint64 landing )
{
   String temp;
   temp.writeNumberHex( landing, true );
   output->writeString( temp );
}


void write_string( Stream *output, uint32 stringId, Module *mod )
{
   String temp;
   if ( ! options.m_inline_str )
   {
      temp += "#";
      temp.writeNumber( (int64) stringId );
      output->writeString( temp );
   }
   else {
      mod->getString( stringId )->escape( temp );
      output->writeString( "\"" + temp + "\"" );
   }
}

void write_symbol( Stream *output, const char *prefix, int32 id )
{
   if ( options.m_isomorphic ) 
   {
      if ( current_symtab != 0 
           && (*prefix == 'L' || *prefix == 'P')  )
      {
         MapIterator iter = current_symtab->map().begin();
         while( iter.hasCurrent() )
         {
            const Symbol*sym = *(const Symbol**) iter.currentValue();
            if ( (sym->isLocal() && *prefix == 'L' && sym->itemId() == id) ||
               (sym->isParam() && *prefix == 'P' && sym->itemId() == id) )
            {
               output->writeString( "$" + sym->name() );
               return;
            }
            iter.next();
         }
      }
      // Search for symbols in the global symbol map
      else if ( *prefix == 'G' )
      {
         MapIterator iter = global_symtab->map().begin();
         while( iter.hasCurrent() )
         {
            const Symbol*sym = *(const Symbol**) iter.currentValue();
            if ( sym->itemId() == id )
            {
               output->writeString( "$" );
               if ( global_symtab != current_symtab )
                  output->writeString( "*" ); //import
               output->writeString( sym->name() );
               return;
            }

            iter.next();
         }
      }
   }

   String temp = "$";
   temp += prefix;
   temp.writeNumber( (int64) id );
   output->writeString( temp );
}



void write_operand( Stream *output, byte *instruction, int opnum, Module *mod )
{
   int offset = 4;
   int count = 1;

   String temp;

   while ( count < opnum ) {
      switch( instruction[ count ] )
      {
         case P_PARAM_NTD32:
         case P_PARAM_INT32:
         case P_PARAM_GLOBID:
         case P_PARAM_LOCID:
         case P_PARAM_PARID:
         case P_PARAM_STRID:
         case P_PARAM_LBIND:
            offset += 4;
         break;

         case P_PARAM_NUM:
         case P_PARAM_INT64:
         case P_PARAM_NTD64:
            offset += 8;
         break;

      }
      count ++;
   }

   switch( instruction[ opnum ] )
   {
      // if it's a NTD32, it may be a jump
      case P_PARAM_NTD32:
         if ( options.m_isomorphic ) {
            byte opcode = *instruction;
            if (
               ( opnum == 1 && ( opcode == P_JMP || opcode == P_TRAL || opcode == P_TRY
                                 || opcode == P_JTRY || opcode == P_IFT || opcode == P_IFF
                                 || opcode == P_ONCE || opcode == P_TRAV
                                 || opcode == P_TRAN || opcode == P_SWCH ) ) ||
               ( opnum == 2 && ( opcode == P_FORK || opcode == P_TRAN ) )
               )
            {
               write_label( output, *((int32*) (instruction + offset )) );
               return;
            }
         }
         temp = "0x";
         temp.writeNumberHex( *((int32*) (instruction + offset )) );
         output->writeString( temp );
      break;


      case P_PARAM_INT32:
         temp.writeNumber( (int64) *((int32*) (instruction + offset )));
         output->writeString( temp );
         break;

      case P_PARAM_NOTUSED: return;

      case P_PARAM_STRID: write_string( output, *((int32*) (instruction + offset )), mod ); break;
      case P_PARAM_LBIND: output->writeString( "$" + *mod->getString( *((int32*) (instruction + offset )))); break;

      case P_PARAM_NUM:
         temp.writeNumber( loadNum( instruction + offset ));
         output->writeString(temp );
         break;
      case P_PARAM_NIL: output->writeString( "NIL" ); break;
      case P_PARAM_TRUE: output->writeString( "T" ); break;
      case P_PARAM_FALSE: output->writeString( "F" ); break;
      case P_PARAM_INT64:
         temp.writeNumber( (int64) loadInt64(instruction + offset) );
         output->writeString(temp );
      break;
      case P_PARAM_GLOBID:
         write_symbol( output, "G", *((int32*) (instruction + offset )) );
      break;
      case P_PARAM_LOCID:
         write_symbol( output, "L", *((int32*) (instruction + offset )) );
      break;
      case P_PARAM_PARID:
         write_symbol( output, "P", *((int32*) (instruction + offset )) );
      break;

      case P_PARAM_NTD64: write_label_long( output, loadInt64((instruction + offset )) ); break;
      case P_PARAM_REGA: output->writeString( "A" ); break;
      case P_PARAM_REGB: output->writeString( "B" ); break;
      case P_PARAM_REGS1: output->writeString( "S1" ); break;
      case P_PARAM_REGL1: output->writeString( "L1" ); break;
      case P_PARAM_REGL2: output->writeString( "L2" ); break;
   }

}

uint32 calc_next( byte *instruction )
{
   uint32 offset = 4;
   int count = 1;
   while ( count < 4 ) {
      switch( instruction[ count ] )
      {
         case P_PARAM_NTD32:
         case P_PARAM_INT32:
         case P_PARAM_GLOBID:
         case P_PARAM_LOCID:
         case P_PARAM_PARID:
         case P_PARAM_STRID:
         case P_PARAM_LBIND:
            offset += 4;
         break;

         case P_PARAM_NUM:
         case P_PARAM_INT64:
         case P_PARAM_NTD64:
            offset += 8;
         break;
      }
      count ++;
   }
   return offset;
}

/***************************************************************
   Isomorphic symbol generators.
*/

void gen_function( Module *module, const Symbol*func, Stream *m_out, t_labelMap labels )
{
   m_out->writeString( "; ---------------------------------------------\n" );
   m_out->writeString( "; Function " + func->name() + "\n" );
   String tempOff;
   tempOff.writeNumberHex( func->getFuncDef()->basePC() );
   m_out->writeString( "; Declared at offset 0x" + tempOff + "\n" );
   m_out->writeString( "; ---------------------------------------------\n\n" );
   m_out->writeString( ".funcdef " + func->name() );
   if ( func->exported() )
      m_out->writeString( " export" );
   m_out->writeString( "\n" );

   // generate the local symbol table.
   const FuncDef *fd = func->getFuncDef();
   MapIterator iter = fd->symtab().map().begin();
   std::vector<Symbol *> params;
   params.resize( fd->symtab().size() );
   while( iter.hasCurrent() )
   {
      const Symbol*sym = *(const Symbol**) iter.currentValue();
      switch( sym->type() ) {
         case Symbol::tparam:
            // paramters must be outputted with their original order.
            params[ sym->itemId() ] = sym;
         break;
         case Symbol::tlocal:
            m_out->writeString( ".local " + sym->name() + "\n" );
         break;

         default:
            break;
      }

      iter.next();
   }

   for ( uint32 parId = 0; parId < params.size(); parId++ )
   {
      const Symbol*sym = params[parId];
      if (sym != 0 )
         m_out->writeString( ".param " + sym->name() + "\n" );
   }

   m_out->writeString( "; ---\n" );

   gen_code( module, fd, m_out, labels );
   m_out->writeString( ".endfunc\n\n" );

}


void gen_propdef( Stream *m_out, const VarDef &def )
{
   String temp;

   switch( def.type() )
   {
      case VarDef::t_nil: m_out->writeString( "NIL" ); break;
      case VarDef::t_bool: m_out->writeString( def.asBool() ? "T": "F" ); break;
      case VarDef::t_int:
         temp.writeNumber( def.asInteger() );
         m_out->writeString( temp );
         break;
      case VarDef::t_num:
         temp.writeNumber( def.asNumeric() );
         m_out->writeString( temp );
         break;
      case VarDef::t_string:
         def.asString()->escape( temp );
         m_out->writeString( "\"" + temp + "\"" );
         break;
      case VarDef::t_reference:
      case VarDef::t_symbol:
         m_out->writeString( "$" +def.asSymbol()->name() );
         break;
      default: break;
   }
}


void gen_class( Stream *m_out, const Symbol*sym )
{
   m_out->writeString( "; ---------------------------------------------\n" );
   m_out->writeString( "; Class " + sym->name() + "\n" );
   m_out->writeString( "; ---------------------------------------------\n\n" );
   m_out->writeString( ".classdef " + sym->name() );
   /*if ( sym->exported() )
      *m_out << " export";*/
   m_out->writeString( "\n" );

   const ClassDef *cd = sym->getClassDef();

   // write class symbol parameters.

   MapIterator st_iter = cd->symtab().map().begin();
   while( st_iter.hasCurrent() )  {
      // we have no locals.
      const String *ptrstr = *(const String **) st_iter.currentKey();
      m_out->writeString( ".param " + *ptrstr +"\n" );
      st_iter.next();
   }

   // write all the inheritances.
   ListElement *it_iter = cd->inheritance().begin();
   while( it_iter != 0 )
   {
      const InheritDef *id = (const InheritDef *) it_iter->data();
      const Symbol*parent = id->base();
      m_out->writeString( ".inherit $" + parent->name() );
      m_out->writeString( "\n" );
      it_iter = it_iter->next();
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

         const String *key = *(const String **) pt_iter.currentKey();
         m_out->writeString( *key + " " );
         gen_propdef( m_out, *def );
         m_out->writeString( "\n" );
      }

      pt_iter.next();
   }

   if ( cd->constructor() != 0 )
   {
      m_out->writeString( ".ctor $" + cd->constructor()->name() + "\n" );
   }

   m_out->writeString( ".endclass\n" );
   m_out->writeString( "; ---------------------------------------------\n" );
}


/**************************************************
   Symbols
*/
void gen_code( Module *module, const FuncDef *fd, Stream *out, const t_labelMap &labels )
{
   uint32 iPos =0;
   byte *code = fd->code();
   byte opcode;
   int oldline = -1;

   // get the main code

   while( iPos < fd->codeSize() )
   {
      opcode = code[ 0 ];

      if ( options.m_isomorphic )
      {

         int l = module->getLineAt( fd->basePC() + iPos );
         if ( l != oldline ) {
            String temp;
            out->writeString( ".line " );
            temp.writeNumber( (int64) l );
            out->writeString( temp + "\n" );
            oldline = l;
         }

         t_labelMap::const_iterator target = labels.find( iPos );
         if ( target != labels.end() ) {
            String temp;
            temp.writeNumber( (int64) target->first );
            out->writeString( "_label_" + temp + ":\n" );
         }
      }
      else {
         String temp;
         temp.writeNumberHex( iPos );
         while( temp.length() < 8 )
            temp = " " + temp;
         out->writeString( temp );
      }
      out->writeString( "\t" );

      const char *csOpName;
      switch( opcode )
      {
         case P_END : csOpName = "END "; break;
         case P_NOP : csOpName = "NOP "; break;
         case P_PSHN: csOpName = "PSHN"; break;
         case P_RET : csOpName = "RET "; break;
         case P_RETA: csOpName = "RETA"; break;

         // Range 2: one parameter ops
         case P_PTRY: csOpName = "PTRY"; break;
         case P_LNIL: csOpName = "LNIL"; break;
         case P_RETV: csOpName = "RETV"; break;
         case P_FORK: csOpName = "FORK"; break;
         case P_BOOL: csOpName = "BOOL"; break;
         case P_GENA: csOpName = "GENA"; break;
         case P_GEND: csOpName = "GEND"; break;
         case P_PUSH: csOpName = "PUSH"; break;
         case P_PSHR: csOpName = "PSHR"; break;
         case P_POP : csOpName = "POP "; break;
         case P_JMP : csOpName = "JMP "; break;
         case P_INC : csOpName = "INC "; break;
         case P_DEC : csOpName = "DEC "; break;
         case P_NEG : csOpName = "NEG "; break;
         case P_NOT : csOpName = "NOT "; break;
         case P_TRAL: csOpName = "TRAL"; break;
         case P_IPOP: csOpName = "IPOP"; break;
         case P_XPOP: csOpName = "XPOP"; break;
         case P_GEOR: csOpName = "GEOR"; break;
         case P_TRY : csOpName = "TRY "; break;
         case P_JTRY: csOpName = "JTRY"; break;
         case P_RIS : csOpName = "RIS "; break;
         case P_BNOT: csOpName = "BNOT"; break;
         case P_NOTS: csOpName = "NOTS"; break;
         case P_PEEK: csOpName = "PEEK"; break;

         // Range3: Double parameter ops
         case P_LD  : csOpName = "LD  "; break;
         case P_LDRF: csOpName = "LDRF"; break;
         case P_ADD : csOpName = "ADD "; break;
         case P_SUB : csOpName = "SUB "; break;
         case P_MUL : csOpName = "MUL "; break;
         case P_DIV : csOpName = "DIV "; break;
         case P_MOD : csOpName = "MOD "; break;
         case P_POW : csOpName = "POW "; break;
         case P_ADDS: csOpName = "ADDS"; break;
         case P_SUBS: csOpName = "SUBS"; break;
         case P_MULS: csOpName = "MULS"; break;
         case P_DIVS: csOpName = "DIVS"; break;
         case P_MODS: csOpName = "MODS"; break;
         case P_POWS: csOpName = "POWS"; break;
         case P_BAND: csOpName = "BAND"; break;
         case P_BOR : csOpName = "BOR "; break;
         case P_BXOR: csOpName = "BXOR"; break;
         case P_ANDS: csOpName = "ANDS"; break;
         case P_ORS : csOpName = "ORS "; break;
         case P_XORS: csOpName = "XORS"; break;
         case P_GENR: csOpName = "GENR"; break;
         case P_EQ  : csOpName = "EQ  "; break;
         case P_NEQ : csOpName = "NEQ "; break;
         case P_GT  : csOpName = "GT  "; break;
         case P_GE  : csOpName = "GE  "; break;
         case P_LT  : csOpName = "LT  "; break;
         case P_LE  : csOpName = "LE  "; break;
         case P_IFT : csOpName = "IFT "; break;
         case P_IFF : csOpName = "IFF "; break;
         case P_CALL: csOpName = "CALL"; break;
         case P_INST: csOpName = "INST"; break;
         case P_ONCE: csOpName = "ONCE"; break;
         case P_LDV : csOpName = "LDV "; break;
         case P_LDP : csOpName = "LDP "; break;
         case P_TRAN: csOpName = "TRAN"; break;
         case P_LDAS: csOpName = "LDAS"; break;
         // when isomorphic, switch is created through directive
         case P_SWCH: csOpName = options.m_isomorphic ? "" : "SWCH"; break;
         case P_IN  : csOpName = "IN  "; break;
         case P_NOIN: csOpName = "NOIN"; break;
         case P_PROV: csOpName = "PROV"; break;
         case P_STPS: csOpName = "STPS"; break;
         case P_STVS: csOpName = "STVS"; break;
         case P_AND : csOpName = "AND "; break;
         case P_OR  : csOpName = "OR  "; break;

         // Range 4: ternary opcodes
         case P_STP : csOpName = "STP "; break;
         case P_STV : csOpName = "STV "; break;
         case P_LDVT: csOpName = "LDVT"; break;
         case P_LDPT: csOpName = "LDPT"; break;
         case P_STPR: csOpName = "STPR"; break;
         case P_STVR: csOpName = "STVR"; break;
         case P_TRAV: csOpName = "TRAV"; break;
         case P_INCP: csOpName = "INCP"; break;
         case P_DECP: csOpName = "DECP"; break;

         case P_SHR : csOpName = "SHR "; break;
         case P_SHL : csOpName = "SHL "; break;
         case P_SHRS: csOpName = "SHRS"; break;
         case P_SHLS: csOpName = "SHLS"; break;
         case P_CLOS: csOpName = "CLOS"; break;
         case P_PSHL: csOpName = "PSHL"; break;
         case P_LSB : csOpName = "LSB "; break;
         case P_SELE: csOpName = options.m_isomorphic  ? "": "SELE"; break;
         case P_INDI: csOpName = "INDI"; break;
         case P_STEX: csOpName = "STEX"; break;
         case P_TRAC: csOpName = "TRAC"; break;
         case P_WRT : csOpName = "WRT "; break;
         case P_STO : csOpName = "STO "; break;
         case P_FORB: csOpName = "FORB"; break;
         case P_EVAL: csOpName = "EVAL"; break;
         case P_OOB : csOpName = "OOB "; break;
         case P_TRDN: csOpName = "TRDN"; break;
         default:
            csOpName = "????";
      }

      out->writeString( csOpName );

      // if the operation is a switch, it's handled a bit specially.
      // now print the operators
      if ( ( opcode != P_SWCH && opcode != P_SELE) || ! options.m_isomorphic )
      {
         if( code[ 1 ] != 0 )
         {
            out->writeString( "\t" );
            write_operand( out, code, 1, module );
            if( code[ 2 ] != 0 )
            {
               out->writeString( ", " );
               write_operand( out, code, 2, module );
               if( code[ 3 ] != 0 )
               {
                  out->writeString( ", " );
                  write_operand( out, code, 3, module );
               }
            }
         }
      }

      // Line marker in non-ismoprphic mode
      if ( options.m_lineinfo && ! options.m_isomorphic )
      {
         int l = module->getLineAt( fd->basePC() + iPos );
         if ( l != oldline ) {
            out->writeString( " ; LINE " );
            String temp;
            temp.writeNumber( (int64) l );
            out->writeString( temp );
            oldline = l;
         }
      }

      //special swch handling
      if ( opcode == P_SWCH || opcode == P_SELE )
      {
         out->writeString( "\n" );
         if ( opcode == P_SWCH )
            out->writeString( ".switch " );
         else
            out->writeString( ".select " );

         write_operand( out, code, 2, module );
         out->writeString( ", " );
         write_operand( out, code, 1, module );
         out->writeString( "\n" );

         uint32 advance = calc_next( code );
         iPos += advance;
         code += advance;

         uint64 sw_count = (uint64) loadInt64( code - sizeof(int64) );

         uint16 sw_int = (int16) (sw_count >> 48);
         uint16 sw_rng = (int16) (sw_count >> 32);
         uint16 sw_str = (int16) (sw_count >> 16);
         uint16 sw_obj = (int16) sw_count;

         //write the nil operand
         if ( *reinterpret_cast<uint32 *>(code) != 0xffffffff )
         {
            if ( ! options.m_isomorphic )
               out->writeString( "\t\t" );
            out->writeString( ".case NIL, " );
            write_label( out, *reinterpret_cast<int32 *>(code) );
            out->writeString( "\n" );
         }
         iPos += sizeof( int32 );
         code += sizeof( int32 );

         // write the integer table
         while( sw_int > 0 )
         {
            if (! options.m_isomorphic )
               out->writeString( "\t\t" );
            out->writeString( ".case " );
            String temp;
            temp.writeNumber( (int64)  loadInt64( code ) );
            out->writeString( temp + ", " );
            code += sizeof( int64 );
            iPos += sizeof( int64 );
            write_label( out, *reinterpret_cast<int32 *>( code ) );
            out->writeString( "\n" );
            code += sizeof( int32 );
            iPos += sizeof( int32 );
            --sw_int;
         }

         // write the range table
         while( sw_rng > 0 )
         {
            if ( !options.m_isomorphic )
              if (! options.m_isomorphic )
               out->writeString( "\t\t" );
            out->writeString( ".case " );
            String temp;
            temp.writeNumber(  (int64)  *reinterpret_cast<int32 *>(code) );
            temp += ":";
            temp.writeNumber(  (int64) *reinterpret_cast<int32 *>(code+ sizeof(int32) ) );
            out->writeString( temp + ", " );
            code += sizeof( int64 );
            iPos += sizeof( int64 );
            write_label( out, *reinterpret_cast<int32 *>( code )  );
            out->writeString( "\n" );
            code += sizeof( int32 );
            iPos += sizeof( int32 );
            --sw_rng;
         }

         // write the string table
         while( sw_str > 0 )
         {
            if ( ! options.m_isomorphic )
               out->writeString( "\t\t" );
            out->writeString( ".case " );
            write_string( out,  *reinterpret_cast<int32 *>(code), module );
            code += sizeof( int32 );
            iPos += sizeof( int32 );
            out->writeString( ", " );
            write_label( out, *reinterpret_cast<int32 *>( code ) );
            out->writeString( "\n" );
            code += sizeof( int32 );
            iPos += sizeof( int32 );
            --sw_str;
         }

         //write the symbol table
         while( sw_obj > 0 )
         {
            if ( !options.m_isomorphic )
               out->writeString( "\t\t" );

            out->writeString( ".case " );
            int32 symId =  *reinterpret_cast<int32 *>(code);
            const Symbol*sym = module->getSymbol( symId );
            if( sym == 0 )
            {
               String temp;
               temp.writeNumber( (int64) symId );
               out->writeString( " [SORRY, SYMBOL " + temp + " NOT FOUND], " );
            }
            else
               out->writeString( "$" + sym->name() + ", " );

            code += sizeof( int32 );
            iPos += sizeof( int32 );
            write_label( out, *reinterpret_cast<int32 *>( code ) );
            out->writeString( "\n" );
            code += sizeof( int32 );
            iPos += sizeof( int32 );
            --sw_obj;
         }

         if ( !options.m_isomorphic )
            out->writeString( "\t" );
         out->writeString( ".endswitch\n\n" );

      }
      else {
         uint32 advance = calc_next( code );
         iPos += advance;
         code += advance;

      }

      out->writeString( "\n" );
   }
}


/********************************************************
   Analizer -- queries the landings of the various
   instructions.
*********************************************************/

void Analizer( FuncDef *fd, t_labelMap &labels )
{
   byte *code, *end;
   byte opcode;

   code = fd->code();
   end = code + fd->codeSize();
   while( code < end )
   {
      opcode = code[ 0 ];

      // check for codes with jump lablels here.
      switch( opcode )
      {
         case P_FORK:
            labels[ *(uint32 *)(code + sizeof(int32)*2 ) ] = 0;
         break;

         case P_TRAN:
            labels[ *(uint32 *)(code + sizeof(int32)*2 ) ] = 0;
         // do not break; fallback

         case P_SWCH:
         case P_JMP :
         case P_TRAL:
         case P_TRY :
         case P_JTRY:
         case P_IFT :
         case P_IFF :
         case P_ONCE:
         case P_TRAV:
            labels[  *(uint32 *)(code + sizeof(int32)) ] = 0;
         break;
      }


      if ( opcode != P_SWCH ) {
         code += calc_next( code );
      }
      else
      {
         //special swch handling
         code += sizeof( int32 ) * 3;  // remove instro and two operands
         uint64 sw_count = (uint64) loadInt64(code);

         uint16 sw_int = (int16) (sw_count >> 48);
         uint16 sw_rng = (int16) (sw_count >> 32);
         uint16 sw_str = (int16) (sw_count >> 16);
         uint16 sw_obj = (int16) sw_count;

         code += sizeof( int64 );

         //write the nil operand
         if ( *reinterpret_cast<uint32 *>(code) != 0xffffffff )
         {
            labels[*reinterpret_cast<int32 *>(code)] = 0;
         }
         code += sizeof( int32 );

         // write the integer table
         while( sw_int > 0 )
         {
            code += sizeof( int64 );
            labels[*reinterpret_cast<int32 *>(code)] = 0;
            code += sizeof( int32 );
            --sw_int;
         }

         // write the range table
         while( sw_rng > 0 )
         {
            code += sizeof( int64 );
            labels[*reinterpret_cast<int32 *>(code)] = 0;
            code += sizeof( int32 );
            --sw_rng;
         }

         // write the string table
         while( sw_str > 0 )
         {
            code += sizeof( int32 );
            labels[*reinterpret_cast<int32 *>(code)] = 0;
            code += sizeof( int32 );
            --sw_str;
         }

         //write the symbol table
         while( sw_obj > 0 )
         {
            code += sizeof( int32 );
            labels[*reinterpret_cast<int32 *>(code)] = 0;
            code += sizeof( int32 );
            --sw_obj;
         }
      }
   }
}

/********************************************************
   Write the string table
*********************************************************/

void write_strtable( e_tabmode mode , Stream *out, Module *mod )
{
   const String *str;
   const StringTable *strtab = &mod->stringTable();

   out->writeString( ";--------------------------------------------\n" );
   out->writeString( "; String table\n" );
   out->writeString( ";--------------------------------------------\n" );
   int32 id = 0;
   while( id < strtab->size() ) {
      str = strtab->get( id );
      String temp;
      switch( mode )
      {
         case e_mode_comment:
            temp.writeNumber( (int64) id );
            out->writeString( "; " + temp + ": " );
            temp.size(0);
         break;
         case e_mode_table:
            out->writeString( str->exported() ? ".istring " : ".string " );
         break;
      }

      str->escape( temp );
      out->writeString( "\"" + temp + "\"\n" );
      ++id;
   }

   out->writeString( "; ------  End of string table -----------\n" );
}


void write_symtable( e_tabmode mode , Stream *out, Module *mod )
{
   const Symbol*sym;
   const SymbolTable *st = &mod->symbolTable();

   out->writeString( ";--------------------------------------------\n" );
   out->writeString( "; Symbol table\n" );
   out->writeString( ";--------------------------------------------\n" );
   MapIterator iter = st->map().begin();

   while( iter.hasCurrent() )
   {
      sym = *(const Symbol**) iter.currentValue();
      if ( sym->name() == "__main__" )
      {
         iter.next();
         continue;
      }

      String temp;
      switch( mode )
      {
         case e_mode_comment:
            out->writeString( "; " );
            switch( sym->type() )
            {
               case Symbol::tundef: out->writeString( "UNDEF" ); break;
               case Symbol::tglobal: out->writeString( "GLOBAL" ); break;
               case Symbol::tlocal: out->writeString( "LOCAL" ); break;
               case Symbol::tparam: out->writeString( "PARAM" ); break;
               case Symbol::tlocalundef: out->writeString( "LU" ); break;
               case Symbol::tfunc:
               {
                  String temp;
                  temp.writeNumberHex( sym->getFuncDef()->basePC(), true );
                  out->writeString( "FUNC(0x" + temp + ")" );
                  break;
               }
               case Symbol::textfunc: out->writeString( "EXTFUNC" ); break;
               case Symbol::tclass: out->writeString( "CLASS" ); break;
               case Symbol::tprop: out->writeString( "PROP" ); break;
               case Symbol::tvar: out->writeString( "VDEF" ); break;
               case Symbol::tinst: out->writeString( "INST" ); break;
               default:
                  break;
            }
            out->writeString( " " + sym->name() + ": " );
            temp.writeNumber( (int64) sym->id() );
            out->writeString( temp );
            temp.size( 0 );
            temp.writeNumber( (int64) sym->declaredAt() );
            out->writeString( " at line "+ temp + " " );
            temp.size( 0 );
            temp.writeNumber( (int64) sym->itemId() );
            out->writeString( "(G"+ temp + ")" );
         break;

         case e_mode_table:
            // see if it's imported
            if ( sym->type() == Symbol::tundef )
            {
               if ( sym->imported() )
               {
                  // see if we have an alias from where to import it
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

               out->writeString( temp );
               break;
            }

            switch( sym->type() )
            {
               case Symbol::tglobal: out->writeString( ".global" ); break;
               case Symbol::tlocal: out->writeString( ".local" ); break;
               case Symbol::tparam: out->writeString( ".param" ); break;
               case Symbol::tfunc: out->writeString( ".func" ); break;
               case Symbol::tclass: out->writeString( ".class" ); break;
               case Symbol::tprop: out->writeString( ".prop" ); break;
               
               case Symbol::tundef: out->writeString( ";---(INVALID) .undef" ); break;
               case Symbol::tlocalundef: out->writeString( ";---(INVALID) .localundef" ); break;
               case Symbol::textfunc: out->writeString( ";---(INVALID) .extfunc" ); break;
               case Symbol::tconst: out->writeString( ";---(INVALID) .const" ); break;
               case Symbol::timportalias: out->writeString( ";---(INVALID) .importalias" ); break;
               case Symbol::tvar: out->writeString( ";---(INVALID) .var" ); break;
               
               case Symbol::tinst:
                  out->writeString( ".instance $" );
                  out->writeString( sym->getInstance()->name() );
                  break;
            }
            out->writeString( " " + sym->name() );
            if( sym->declaredAt() != 0 ) {
               String temp = " ";
               temp.writeNumber( (int64) sym->declaredAt() );
               out->writeString( temp );
            }
         break;
      }
      if ( sym->exported() )
         out->writeString( " export" );
      out->writeString( "\n" );

      iter.next();
   }

   out->writeString( "; ------  End of symbol table ---------------\n" );
}


void write_deptable( e_tabmode mode , Stream *out, Module *mod )
{
   const String *str;
   const DependTable *deps = &mod->dependencies();
   current_symtab = 0;

   out->writeString( ";--------------------------------------------\n" );
   out->writeString( "; Dependencies table\n" );
   out->writeString( ";--------------------------------------------\n" );

   MapIterator iter = deps->begin();

   while( iter.hasCurrent() )
   {
      const String *alias = *(const String **) iter.currentKey();
      const ModuleDepData *data = *(const ModuleDepData **) iter.currentValue();
      str = &data->moduleName();

      if ( data->isPrivate() )
      {
         if ( mode == e_mode_comment )
         {
            out->writeString( ";" );

            if ( data->isFile() )
               out->writeString( " \"" + *str + "\"" );
            else
               out->writeString( " " + *str );

            if ( *str != *alias )
               out->writeString( " -> " + *alias );

            out->writeString( " (private)" );
         }
      }
      else {
         switch( mode )
         {
            case e_mode_comment:
               out->writeString( ";" );
            break;
            case e_mode_table:
               out->writeString( ".load" );
            break;
         }

         if ( data->isFile() )
            out->writeString( " \"" + *str + "\"" );
         else
            out->writeString( " " + *str );
      }

      out->writeString( "\n" );
      iter.next();
   }

   out->writeString( "; ------  End of depend table -----------\n" );
}


void usage()
{
   ::Falcon::Stream *stdOut = ::Falcon::stdOutputStream();
   stdOut->writeString( "USAGE: faldisass [options] [filename]\n" );
   stdOut->writeString( "\t-s: Dump the string table\n" );
   stdOut->writeString( "\t-S: Write the strings inline instead of using #strid\n" );
   stdOut->writeString( "\t-y: Dump the symbol table\n" );
   stdOut->writeString( "\t-d: Dump the dependency table\n" );
   stdOut->writeString( "\t-l: add line informations\n" );
   stdOut->writeString( "\t-h: show this help\n" );
   stdOut->writeString( "\t-i: create an isomorphic version of the original assembly\n" );
   stdOut->writeString( "\tIf 'filename' is '-' or missing, read from stdin\n" );
   stdOut->flush();
}

}

using namespace Falcon;

int main( int argc, char *argv[] )
{
   // initialize the engine
   Falcon::Engine::AutoInit autoInit;

   Stream *stdErr = stdErrorStream();

   // option decoding
   for ( int i = 1; i < argc; i++ )
   {
      char *option = argv[i];
      if (option[0] == '-' )
      {
         if ( option[1] == 0 ) {
            options.m_stdin = true;
            options.m_fname = 0;
            continue;
         }
         int pos = 1;
         while ( option[pos] != 0 )
         {
            switch ( option[ pos ] )
            {
               case 'i': options.m_isomorphic = true; options.m_inline_str = true; break;
               case 's': options.m_dump_str = true; break;
               case 'l': options.m_lineinfo = true; break;
               case 'S': options.m_inline_str = true; break;
               case 'y': options.m_dump_sym = true; break;
               case 'd': options.m_dump_dep = true; break;
               case 'h': case '?': usage(); return 0; break;
               default:
                  stdErr->writeString( "faldisass: unrecognized option '-" );
                  stdErr->put( option[pos] );
                  stdErr->writeString( "'.\n" );
                  usage();
                  stdErr->flush();
                  return 1;
            }
            ++pos;
         }
      }
      else {
         options.m_fname = option;
         options.m_stdin = false;
      }
   }

   Stream *in;
   // file opening
   FileStream inf;
   if( options.m_fname != 0 && ! options.m_stdin )
   {
      inf.open( options.m_fname );
      if ( ! inf.good() )
      {
         stdErr->writeString( "faldisass: can't open file " );
         stdErr->writeString( options.m_fname );
         stdErr->writeString( "\n" );
         stdErr->flush();
         return 1;
      }
      in = &inf;
   }
   else {
      in = stdInputStream();
   }


   Module *module = new Module();

   if( ! module->load( in, false ) )
   {
      if ( options.m_fname == 0 || options.m_stdin )
         stdErr->writeString( "faldisass: invalid module in input stream\n" );
      else {
         stdErr->writeString( "faldisass: invalid module " );
         stdErr->writeString( options.m_fname );
         stdErr->writeString( "\n" );
      }
      stdErr->flush();
      return 1;
   }

   global_symtab = &module->symbolTable();

   // for isomorphic we need the analysis.
   Stream *stdOut = DefaultTextTranscoder( new StdOutStream );
   t_labelMap labels;
   if ( options.m_isomorphic ) {

      // print immediately the string table.
      write_strtable( e_mode_table, stdOut, module );

      // and then the load table.
      write_deptable( e_mode_table, stdOut, module );

      // and the symbol table.
      write_symtable( e_mode_table, stdOut, module );
   }

   // now write MAIN, if it exists
   const Symbol*main = module->findGlobalSymbol( "__main__" );
   if ( main != 0 )
   {
      current_symtab = global_symtab;
      if ( options.m_isomorphic )
      {
         labels.clear();
         Analizer( main->getFuncDef(), labels );
      }
      gen_code( module, main->getFuncDef(), stdOut, labels );
   }

   // also fetch the function map.
   MapIterator iter = module->symbolTable().map().begin();

   while( iter.hasCurrent() )
   {
      const Symbol*sym = *(const Symbol**) iter.currentValue();
      if ( sym->name() == "__main__" )
      {
         // already managed
         iter.next();
         continue;
      }

      if ( sym->isFunction() )
      {
         current_symtab = &sym->getFuncDef()->symtab();

         if ( options.m_isomorphic )
         {
            labels.clear();
            Analizer( sym->getFuncDef(), labels );
            gen_function( module, sym, stdOut, labels );
         }
         else {
            String tempOff;
            tempOff.writeNumberHex( sym->getFuncDef()->basePC() );
            stdOut->writeString( "\n; Generating function " + sym->name() + "(0x"+ tempOff + ")\n" );
            gen_code( module, sym->getFuncDef(), stdOut, labels );
         }


      }
      else if ( sym->isClass() && options.m_isomorphic ) {
         // we can generate the classess now, as the symbols have been
         // already declared.
         gen_class( stdOut, sym );
      }

      iter.next();
   }

   if ( options.m_dump_str ) {
      write_strtable( e_mode_comment, stdOut, module );
   }
   if ( options.m_dump_sym ) {
      write_symtable( e_mode_comment, stdOut, module );
   }
   if ( options.m_dump_dep ) {
      write_deptable( e_mode_comment, stdOut, module );
   }

   stdOut->writeString( "\n" );
   module->decref();

   return 0;
}


/* end of faldisass.cpp */
