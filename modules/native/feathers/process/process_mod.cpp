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


namespace Falcon { namespace Mod {


void argvize(GenericVector& argv, const String &params)
{
   typedef enum {
      s_none,
      s_quote1,
      s_quote2,
      s_escape1,
      s_escape2,
      s_token
   } t_state;

   t_state state = s_none;

   // string lenght
   uint32 len = params.length();
   uint32 pos = 0;
   uint32 posInit = 0;

   if( len > 0 )
   {
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
                     argv.push(new String( params, posInit, pos ));
                     state = s_none;
                  break;

                  // In case of " change state but don't change start position
                  case '\"':
                     argv.push(new String( params, posInit, pos ));
                     posInit = pos + 1;
                     state = s_quote1;
                  break;

                  case '\'':
                     argv.push(new String( params, posInit, pos ));                     
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
                  argv.push(new String( params, posInit, pos ));
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
                  argv.push(new String( params, posInit, pos ));
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
      argv.push(new String( params, posInit, pos ));
}

}} // ns Falcon::Mod


/* end of process_mod.cpp */
