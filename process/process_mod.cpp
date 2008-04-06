/*
   FALCON - The Falcon Programming Language.
   FILE: process_mod.cpp

   Process module common functions and utilities.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab mar 11 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Process module common functions and utilities.
*/

#include <ctype.h>

#include <falcon/memory.h>
#include <falcon/string.h>
#include <string.h>

#include "process_sys.h"
#include "process_mod.h"


namespace Falcon {

namespace Mod {

const int assing_block = 32;

static String **assignString( const String &params, String **args, uint32 &assigned, uint32 &count, uint32 posInit, uint32 pos )
{
   if( assigned == count ) {
      assigned += assing_block;
      String **temp = (String **) memAlloc( assigned * sizeof( String * ) );
      if ( assigned > assing_block )
         memcpy( temp, args, (assigned - assing_block) * sizeof( String * ) );
      memFree( args );
      args = temp;
   }
   args[ count++ ] = new String( params, posInit, pos );
   return args;
}

String **argvize( const String &params, bool addShell )
{
   typedef enum {
      s_none,
      s_quote1,
      s_quote2,
      s_escape1,
      s_escape2,
      s_token
   } t_state;

   t_state state;

   uint32 start;
   if ( addShell )
      start = 2;
   else
      start = 0;

   // string lenght
   uint32 len = params.length();
   uint32 pos = 0;
   uint32 posInit = 0;
   uint32 count = 0;
   String **args;
   uint32 assigned = assing_block;
   args = (String **) memAlloc( assigned * sizeof( String * ) );

   if( len > 0 )
   {
      state = s_none;
      while( pos < len )
      {
         uint32 chr = params.getCharAt( pos );

         switch( state )
         {
            case s_none:
               switch( chr )
               {
                  case ' ': case '\t':
                  break;

                  case '\"':
                     state = s_quote1;
                     posInit = pos;
                  break;

                  case '\'':
                     state = s_quote2;
                     posInit = pos;
                  break;

                  default:
                     state = s_token;
                     posInit = pos;
               }
            break;

            case s_token:
               switch( chr )
               {
                  case ' ': case '\t':
                     args = assignString( params, args, assigned, count, posInit, pos );
                     state = s_none;
                  break;

                  // In case of " change state but don't change start position
                  case '\"':
                     args = assignString( params, args, assigned, count, posInit, pos );
                     posInit = pos + 1;
                     state = s_quote1;
                  break;

                  case '\'':
                     args = assignString( params, args, assigned, count, posInit, pos );
                     posInit = pos + 1;
                     state = s_quote2;
                  break;
               }
            break;

            case s_quote1:
               if ( chr == '\\' )
                  state = s_escape1;
               else if ( chr == '\"' )
               {
                  args = assignString( params, args, assigned, count, posInit, pos );
                  state = s_none;
               }
            break;

            case s_escape1:
               state = s_quote1;
            break;

            case s_quote2:
               if ( chr == '\\' )
                  state = s_escape2;
               else if ( chr == '\'' )
               {
                  args = assignString( params, args, assigned, count, posInit, pos );
                  state = s_none;
               }
            break;

            case s_escape2:
               state = s_quote2;
            break;
         }

         pos ++;
      }
   }

   // last
   if( state != s_none && posInit < pos )
   {
      args = assignString( params, args, assigned, count, posInit, pos );
   }
   args[ count ] = 0;
   return args;
}

void freeArgv( String **argv )
{
   String **p = argv;
   while( *p != 0 )
   {
      delete *p;
      ++p;
   }
   memFree( argv );
}

/*
int parametrize( char *out, const String &params )
{
   int count = 0;  // we'll have at least one token

   // removes leading spaces
   while ( *in && isspace(*in) )
      in++;
   if (! *in ) return 0;

   while ( *in ) {
      if ( *in == '\"' || *in == '\'')
      {
         char quote = *in;
         in++;
         while ( *in && *in != quote ) {
            if ( *in == '\\' ) {
               in++;
            }
            if ( *in ) {
               *out = *in;
               out ++;
               in++;
            }
         }
         if (*in) {
            in++;
         }
         if ( *in ) {
            *out = '\0';
         }
         // out++ will be done later; if in is done,
         // '\0' will be added at loop exit.
      }
      else if (! isspace( *in ) ) {
         *out = *in;
         in++;
         out++;
      }
      else {
         *out = '\0';
         count ++;
         while (*in && isspace( *in ) )
            in++;
         out++;
      }
   }
   *out = '\0';
   count ++;

   return count;
}
*/
}
}


/* end of process_mod.cpp */
