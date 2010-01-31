/*
   FALCON - The Falcon Programming Language.
   FILE: options.cpp

   Options for falcon packager - implementation.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 30 Jan 2010 12:42:48 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "options.h"
#include "utils.h"

#include <cstring>

namespace Falcon
{

Options::Options():
      m_bPackFam( false ),
      m_bStripSources( false ),
      m_bNoSysFile( false ),
      m_sSystemRoot( "_system" ),
      m_bHelp( false ),
      m_bVersion( false ),
      m_bVerbose( false ),
      m_bIsValid( true )
{
   m_sysModules.insert( "compiler" );
   m_sysModules.insert( "funcext" );
   m_sysModules.insert( "confparser" );
   m_sysModules.insert( "json" );
   m_sysModules.insert( "logger" );
   m_sysModules.insert( "mxml" );
   m_sysModules.insert( "regex" );
   m_sysModules.insert( "process" );
   m_sysModules.insert( "socket" );
   m_sysModules.insert( "threading" );
   m_sysModules.insert( "zlib" );

}

bool Options::parse( int argc, char* const argv[] )
{
   int p = 0;
   String* getMe = 0;
   String blackListItem;

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

         if( &blackListItem == getMe )
         {
            m_blackList.insert( blackListItem );
         }

         getMe = 0;
      }
      else if( word[0] == '-' )
      {
         switch( word[1] )
         {
         case 'M': m_bPackFam = true; break;
         case 's': m_bStripSources = true; break;
         case 'S': m_bNoSysFile = true; break;
         case '?': case 'h': m_bHelp = true; break;
         case 'v': m_bVersion = true; break;
         case 'V': m_bVerbose = true; break;

         case 'b': getMe = &blackListItem; break;
         case 'e': getMe = &m_sEncoding; break;
         case 'P': getMe = &m_sTargetDir; break;
         case 'L': getMe = &m_sLoadPath; break;
         case 'r': getMe = &m_sRunner; break;
         case 'R': getMe = &m_sSystemRoot; break;

         case '-':
            if( strcmp( word + 2, "bin" ) == 0 )
            {
               getMe = &m_sFalconBinDir;
               break;
            }
            else if( strcmp( word + 2, "lib" ) == 0 )
            {
               getMe = &m_sFalconLibDir;
               break;
            }
            // else fallthrough and raise error

         default:
            error( String("Invalid option \"").A(word).A("\"") );
            m_bIsValid = false;
            return false;
         }
      }
      else
      {
         if ( m_sMainScript !=  "" )
         {
            // but it's not an error -- main will tell "nothing to do" and exit.
            m_bIsValid = false;
            return false;
         }

         m_sMainScript.bufferize(word);
      }
      ++p;
   }

   // do we miss the last parameter?
   if ( getMe != 0 )
   {
      error( String("Option \"").A(argv[ argc-1 ]).A("\" needs a parameter.") );
      m_bIsValid = false;
      return false;
   }

   // do we miss both sources and fams?
   if( m_bStripSources && ! m_bPackFam )
   {
      error( String("Options -M and -s are incompatible") );
      m_bIsValid = false;
      return false;
   }

   return true;
}

bool Options::isBlackListed( const String& modname ) const
{
   return m_blackList.find( modname ) != m_blackList.end();
}

bool Options::isSysModule( const String& modname ) const
{
   return m_sysModules.find( modname ) != m_sysModules.end();
}


}

/* end of options.cpp */
