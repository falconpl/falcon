/*
   FALCON - The Falcon Programming Language.
   FILE: pcode.cpp

   Utilities to manage the Falcon Virtual Machine pseudo code.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 24 Jul 2009 19:42:40 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/pcode.h>
#include <falcon/common.h>
#include <falcon/module.h>
#include <falcon/symbol.h>

namespace Falcon {


void PCODE::convertEndianity( uint32 paramType, byte* targetArea )
{
   switch( paramType )
   {
      case P_PARAM_INT32:
      case P_PARAM_STRID:
      case P_PARAM_LBIND:
      case P_PARAM_GLOBID:
      case P_PARAM_LOCID:
      case P_PARAM_PARID:
      case P_PARAM_NTD32:
         *reinterpret_cast<int32 *>(targetArea) = endianInt32(*reinterpret_cast<int32 *>(targetArea) );
         break;

      case P_PARAM_INT64:
      case P_PARAM_NTD64:
      {
         uint64 value64 = grabInt64( targetArea );

         // high part - low part
         *reinterpret_cast<uint32 *>(targetArea) = value64 >> 32;
         *reinterpret_cast<uint32 *>(targetArea+sizeof(uint32)) = (uint32) value64;
      }
      break;

      case P_PARAM_NUM:
      {
         union t_unumeric {
            struct t_integer {
                 uint32 high;
                 uint32 low;
              } integer;
              numeric number;
         } unumeric;

         unumeric.number = grabNum( targetArea );

         // high part - low part
         *reinterpret_cast<uint32 *>(targetArea) = unumeric.integer.high;
         *reinterpret_cast<uint32 *>(targetArea+sizeof(uint32)) = unumeric.integer.low;
      }
      break;
   }

}


uint32 PCODE::advanceParam( uint32 paramType )
{
   uint32 offset;

   switch( paramType )
   {

   case P_PARAM_NIL:
   case P_PARAM_NOTUSED:
   case P_PARAM_TRUE:
   case P_PARAM_FALSE:
   case P_PARAM_REGA:
   case P_PARAM_REGB:
   case P_PARAM_REGS1:
   case P_PARAM_REGL1:
   case P_PARAM_REGL2:
      offset = 0;
      break;

   case P_PARAM_NUM:
   case P_PARAM_INT64:
   case P_PARAM_NTD64:
      offset = 8;
      break;

   default:
      offset = 4;
   }

   return offset;
}


void PCODE::deendianize( byte* code, uint32 codeSize )
{
   uint32 iPos =0;
   byte opcode;
   while( iPos < codeSize )
   {
      opcode = code[ iPos ];
      uint32 iStart = iPos;
      // get the options
      iPos += 4;
      if( code[ iStart + 1 ] != 0 )
      {
         convertEndianity( code[ iStart + 1 ], code + iPos );
         iPos += advanceParam( code[iStart + 1] );

         if( code[ iStart + 2 ] != 0 )
         {
            convertEndianity( code[iStart + 2], code + iPos );
            iPos += advanceParam( code[iStart + 2] );

            if( code[ iStart + 3 ] != 0 )
            {
               convertEndianity( code[iStart + 3], code + iPos );
               iPos += advanceParam( code[iStart + 3] );
            }
         }
      }

      // if the operation is a switch, it's handled a bit specially.
      if ( opcode == P_SWCH || opcode == P_SELE )
      {
         // get the switch table (aready de-endianized)
         uint64 sw_count = *reinterpret_cast<uint64 *>(code + iPos - sizeof(int64));

         uint16 sw_int = (int16) (sw_count >> 48);
         uint16 sw_rng = (int16) (sw_count >> 32);
         uint16 sw_str = (int16) (sw_count >> 16);
         uint16 sw_obj = (int16) sw_count;

         // Endianize the nil landing
         *reinterpret_cast<uint32 *>(code+iPos) = endianInt32( *reinterpret_cast<uint32 *>(code+iPos) );
         iPos += sizeof( uint32 );

         // endianize the integer table
         while( sw_int > 0 )
         {
            // the int64 value
            uint64 value64 = grabInt64( code+iPos );
            // high part - low part
            *reinterpret_cast<uint32 *>(code+iPos) = value64 >> 32;
            *reinterpret_cast<uint32 *>(code+iPos+sizeof(uint32)) = (uint32) value64;
            iPos += sizeof( int64 );
            // and the landing
            *reinterpret_cast<int32 *>(code+iPos) = endianInt32( *reinterpret_cast<int32 *>(code+iPos) );
            iPos += sizeof( int32 );

            --sw_int;
         }

         // endianize the range table
         while( sw_rng > 0 )
         {
              // the int32 value -- start
              *reinterpret_cast<int32 *>(code+iPos) = endianInt32( *reinterpret_cast<int32 *>(code+iPos) );
              iPos += sizeof( int32 );
              // the int32 value -- end
              *reinterpret_cast<int32 *>(code+iPos) = endianInt32( *reinterpret_cast<int32 *>(code+iPos) );
              iPos += sizeof( int32 );

              // and the landing
              *reinterpret_cast<int32 *>(code+iPos) = endianInt32( *reinterpret_cast<int32 *>(code+iPos) );
              iPos += sizeof( int32 );

              --sw_rng;
         }

         // endianize the string table
         while( sw_str > 0 )
         {
              // the int32 string index
              *reinterpret_cast<int32 *>(code+iPos) = endianInt32( *reinterpret_cast<int32 *>(code+iPos) );
              iPos += sizeof( int32 );

              // and the landing
              *reinterpret_cast<int32 *>(code+iPos) = endianInt32( *reinterpret_cast<int32 *>(code+iPos) );
              iPos += sizeof( int32 );

              --sw_str;
         }

         // endianize the object table
         while( sw_obj > 0 )
         {
              // the int32 symbol index
              *reinterpret_cast<int32 *>(code+iPos) = endianInt32( *reinterpret_cast<int32 *>(code+iPos) );
              iPos += sizeof( int32 );

              // and the landing
              *reinterpret_cast<int32 *>(code+iPos) = endianInt32( *reinterpret_cast<int32 *>(code+iPos) );
              iPos += sizeof( int32 );

              --sw_obj;
         }
      }

   }
}


void PCODE::deendianize( Module* mod )
{
   const SymbolTable &symtab = mod->symbolTable();


   // now, the symbol table must be traversed.
   MapIterator iter = symtab.map().begin();
   while( iter.hasCurrent() )
   {
      Symbol *sym = *(Symbol **) iter.currentValue();

      if ( sym->isFunction() )
      {
         FuncDef* fd = sym->getFuncDef();
         deendianize( fd->code(), fd->codeSize() );
      }

      // next symbol
      iter.next();
   }
}

}

/* end of pcode.cpp */
