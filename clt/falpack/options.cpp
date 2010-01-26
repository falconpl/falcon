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
      m_sEncoding(""),
      m_bHelp( false ),
      m_bVersion( false ),
      m_bIsValid( true )
{
}

bool Options::parse( int argc, char* const argv[] )
{
   int p = 0;
   String* getMe = 0;

   while( p < argc )
   {
      const char* word = argv[ p ];
      if ( word == 0 )
      {
         // !?!?!? Malformed argv?
         m_bIsValid = false;
         return false;
      }

      if ( getMe != 0 )
      {
         getMe->bufferize( word );
         getMe = 0;
      }
      else if( word[0] == '-' )
      {
         switch( word[1] )
         {
         case 'M': m_bPackFam = true; break;
         case 's': m_bStripSources = true; break;
         case 'S': m_bNoSysFile = true; break;
         case 'r': m_bUseFalrun = true; break;
         case '?': case 'h': m_bHelp = true; break;
         case 'v': m_bVersion = true; break;

         case 'e':  getMe = &m_sEncoding; break;
         case 'P':  getMe = &m_sTargetDir; break;
         case 'L':  getMe = &m_sLoadPath; break;

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

   // do we miss the last parameter?
   if ( getMe != 0 )
   {
      m_bIsValid = false;
      return false;
   }

   return true;
}

}

/* end of options.cpp */
