/*
   The Falcon Programming Language
   FILE: dynlib_mod.cpp

   Direct dynamic library interface for Falcon
   Interface extension functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 28 Oct 2008 22:23:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: The Falcon Comittee

   See the LICENSE file distributed with this package for licensing details.
*/

/** \file
   Direct dynamic library interface for Falcon
   Internal logic functions - implementation.
*/

#include "dynlib_mod.h"

#include <stdio.h>

namespace Falcon {

FunctionAddress::~FunctionAddress()
{
   delete[] m_parsedParams;
   delete[] m_safetyParams;
}

void FunctionAddress::gcMark( VMachine * )
{
   // nothing to mark
}

FalconData *FunctionAddress::clone() const
{
   return new FunctionAddress( *this );
}

bool FunctionAddress::parseParams( const String &mask )
{
   m_bGuessParams = false; // we know we have some paramters to parse.

   // delete in case we had some.
   delete[] m_parsedParams;
   delete[] m_safetyParams;

   // allocate a sensible amount of data.
   uint32 plen = mask.length();

   if ( plen == 0 )
   {
      // it's just a declaration of "no parameters"
      m_parsedParams = new byte[1];
      m_parsedParams[0] = F_DYNLIB_PTYPE_END;
      return true;
   }

   if( plen < F_DYNLIB_MAX_PARAMS )
   {
      // we're pretty sure that the size can't be larger; it's useless to allocate more.
      m_parsedParams = new byte[plen+1];
   }
   else {
      m_parsedParams = new byte[F_DYNLIB_MAX_PARAMS+1];
   }

   // we need some pairs for later creation of safety params.
   uint32 safeStarts[F_DYNLIB_MAX_PARAMS];
   uint32 safeEnds[F_DYNLIB_MAX_PARAMS];
   uint32 safeCount = 0;

   // ok, we can proceed in parsing.
   uint32 pos = 0;
   uint32 spos = 0;
   uint32 parsedTokens = 0;
   bool empty = true;

   while( pos <= plen && parsedTokens < F_DYNLIB_MAX_PARAMS )
   {
      // find the element between whitespaces
      uint32 chr;
      if ( pos < plen )
         chr = mask.getCharAt( pos );
      else
         chr = ' '; // fake an extra space at the end.

      if ( chr == ' ' || chr == ',' || chr == ';' )
      {
         if ( ! empty )
         {
            // we found the limit of a previous token
            byte token;
            if ( ! parseSingleParam( mask, token, spos, pos ) )
               return false;

            // ok the token was valid - was it a safety string?
            if( token == F_DYNLIB_PTYPE_OPAQUE )
            {
               // record the string.
               safeStarts[safeCount] = spos;
               safeEnds[safeCount] = pos;
               ++safeCount;
            }
            else if ( token == (F_DYNLIB_PTYPE_OPAQUE | F_DYNLIB_PTYPE_BYPTR) )
            {
               // record the string.
               safeStarts[safeCount] = spos+1;
               safeEnds[safeCount] = pos;
               ++safeCount;
            }
            // was it ... -- then the string must be over NOW
            else if ( token == F_DYNLIB_PTYPE_VAR )
            {
               if( pos != plen )
                  return false;
            }

            // anyhow accept the token and move on.
            m_parsedParams[parsedTokens++] = token;
            empty = true;
         }
         // else, just ignore.
      }
      else {
         if ( empty )
         {
            // start a new token.
            empty = false;
            spos = pos;
         }
         // otherwise just ignore.
      }

      ++pos;  // advance
   }

   printf( "Parsing complete: %d tokens\n", parsedTokens );

   // Exited because of excessive tokens?
   if( pos < plen )
      return false;

   // close the sequence
   m_parsedParams[ parsedTokens ] = F_DYNLIB_PTYPE_END;

   // Now create the string vector containing our safety types
   if( safeCount > 0 )
   {
      m_safetyParams = new String[safeCount];
      for( int i = 0; i < safeCount; i++ )
      {
         m_safetyParams[i] = mask.subString(safeStarts[i], safeEnds[i]);
      }
   }

   return true;
}


bool FunctionAddress::parseSingleParam( const String &mask, byte &type, uint32 begin, uint32 end )
{
   uint32 pos = begin;
   byte prefix = 0;
   byte value = 0;

   typedef enum  {
      es_begin,
      es_firstchar,
      es_symbol,
      es_firstdot,
      es_seconddot,
      es_thirddot,
      es_maybesym,
      es_invalid
   } t_state;

   t_state state = es_begin;

   if ( end > mask.length() )
      end = mask.length();

   while ( pos < end )
   {
      uint32 chr = mask.getCharAt( pos );

      switch( state )
      {
         case es_begin:
            switch( chr )
            {
               case '$':
                  if (value != 0 || prefix != 0 )
                  {
                     // this wasn't the first character.
                     return false;
                  }

                  prefix = F_DYNLIB_PTYPE_BYPTR;
                  // but don't change the state.
                  break;

               case 'P':
                  value = F_DYNLIB_PTYPE_PTR;
                  state = es_firstchar;
                  break;

               case 'F':
                  value = F_DYNLIB_PTYPE_FLOAT;
                  state = es_firstchar;
                  break;

               case 'D':
                  value = F_DYNLIB_PTYPE_DOUBLE;
                  state = es_firstchar;
                  break;

               case 'I':
                  value = F_DYNLIB_PTYPE_I32;
                  state = es_firstchar;
                  break;

               case 'U':
                  value = F_DYNLIB_PTYPE_U32;
                  state = es_firstchar;
                  break;

               case 'L':
                  value = F_DYNLIB_PTYPE_LI;
                  state = es_firstchar;
                  break;

               case 'S':
                  value = F_DYNLIB_PTYPE_SZ;
                  state = es_firstchar;
                  break;

               case 'W':
                  value = F_DYNLIB_PTYPE_WZ;
                  state = es_firstchar;
                  break;

               case 'M':
                  value = F_DYNLIB_PTYPE_MB;
                  state = es_firstchar;
                  break;

               case '.':
                  state = es_firstdot;
                  break;

               default:
                  if( chr > 255 )
                  {
                     // a wide char - counts as a symbol
                     state = es_symbol;
                     value = F_DYNLIB_PTYPE_OPAQUE;
                  }
                  else {
                     state = es_maybesym;
                  }
            }
            break;

         case es_maybesym:
         case es_firstchar:
            // well, we have a char, so
            if ( chr == '$' || chr == '.' )
               return false;
            else {
               state = es_symbol;
               value = F_DYNLIB_PTYPE_OPAQUE;
            }
            break;

         case es_firstdot:
            // well, we have a char, so
            if ( chr != '.' )
               return false;
            else
               state = es_seconddot;
            break;

         case es_seconddot:
            // well, we have a char, so
            if ( chr != '.' )
               return false;
            else
               value = F_DYNLIB_PTYPE_VAR;
               state = es_thirddot;
            break;

          case es_thirddot:
            // nothing should be after a third dot
            return false;

         case es_symbol:
            if ( chr < 256 &&
               ( (chr|0x32) < 'a' || (chr|0x32) > 'z' )
               )
               return false;
            break;
      }

      ++pos;
   }

   if( state == es_maybesym )
   {
      // a single character...
      return false;
   }

   // we are at the end of input, and all is fine,
   type = prefix | value;
   return true;
}


bool FunctionAddress::parseReturn( const String &rval )
{
   m_returnMask = rval;
   m_returnMask.trim();

   return parseSingleParam( m_returnMask, m_parsedReturn );
}


/*
void FunctionAddress::call( VMachine *vm, int32 firstParam ) const
{
}
*/

//===========================================
// DynFunc manager
//
bool DynFuncManager::isFalconData() const
{
   return true;
}

void *DynFuncManager::onInit( Falcon::VMachine * )
{
   return 0;
}

void DynFuncManager::onDestroy( Falcon::VMachine *, void *user_data )
{
   FunctionAddress *fa = reinterpret_cast<FunctionAddress *>( user_data );
   delete fa;
}

void *DynFuncManager::onClone( Falcon::VMachine *, void *user_data )
{
   FunctionAddress *fa = reinterpret_cast<FunctionAddress *>( user_data );
   return fa->clone();
}


}


/* end of dynlib_mod.cpp */
