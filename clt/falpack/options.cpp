/*
   FALCON - The Falcon Programming Language.
   FILE: options.cpp

   Falcon compiler and interpreter
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "options.h"

namespace Falcon
{

Options::Options():
      m_bPackFam( false ),
      m_bStripSources( false ),
      m_bNoSysFile( false ),
      m_bUseFalrun( false ),
      m_sTargetDir( "" ),
      m_sLoadPath( "" ),
      m_sMainScript( "" ),
      m_bIsValid( true )
{
}

bool Options::parse( int argc, char* const argv[] )
{
   int p = 0;
   while( p < argc )
   {
      const char* word = argv[ p ];
      if ( word == 0 )
      {
         // !?!?!? Malformed argv?
         m_bIsValid = false;
         return false;
      }

      if( word[0] == '-' )
      {
         switch( word[1] )
         {
         case 'M': m_bPackFam = true; break;
         case 's': m_bStripSources = true; break;
         case 'S': m_bNoSysFile = true; break;
         case 'r': m_bUseFalrun = true; break;
         
         case 'P': /* TODO */ ; break;
         case 'L': /* TODO */ ; break;

         default:
            m_bIsValid = false;
            return false;
         }
      }
      else
      {
         if ( m_sMainScript !=  "" )
         {
            m_bIsValid = false;
            return false;
         }

         m_sMainScript.bufferize(word);
      }
   }

   return true;
}

}

/* end of options.cpp */
