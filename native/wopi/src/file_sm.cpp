/*
   FALCON - The Falcon Programming Language.
   FILE: file_sm.cpp

   Falcon Web Oriented Programming Interface

   File based session manager.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 27 Mar 2010 15:04:36 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/wopi/file_sm.h>

#include <falcon/sys.h>
#include <falcon/filestat.h>
#include <falcon/fstream.h>
#include <falcon/item.h>

#include <falcon/wopi/request.h>

#define FALCON_SESSION_DATA_EXT ".fsd"


namespace Falcon {
namespace WOPI {

//============================================================
// File based session data
//

FileSessionData::FileSessionData( const String& SID, const String& tmpDir ):
   SessionData( SID ),
   m_tmpDir( tmpDir )
{
}

FileSessionData::~FileSessionData()
{
}

bool FileSessionData::resume()
{
   String sFile = m_tmpDir + "/" + sID() + FALCON_SESSION_DATA_EXT;

   FileStat fst;
   if ( ! Sys::fal_stats( sFile, fst ) )
   {
      // is this a new session, or is the directory plainly wrong?
      if ( (! Sys::fal_stats( m_tmpDir, fst )) || fst.m_type == FileStat::t_normal )
      {
         setError( "Not a directory " + m_tmpDir );
         return false;
      }

      // it's a new session!
      return false;
   }
   else
   {
      // deserialize from this file.
      FileStream fs;
      if ( ! fs.open( sFile, FileStream::e_omReadOnly ) )
      {
         setError( "Can't open session file " + sFile );
         return false;
      }

      // don't use a VM to deserialize
      if( m_dataLock.item().deserialize( &fs, VMachine::getCurrent() ) != Item::sc_ok )
      {
         setError( "Deserialization failed from " + sFile );

         // but let the user see the error.
         return false;
      }
   }

   return true;
}


bool FileSessionData::store()
{
   FileStream fs;
   String sFile = m_tmpDir + "/" + sID() + FALCON_SESSION_DATA_EXT;

   // try to create the file.
   if ( ! fs.create( sFile, FileStream::e_aUserWrite | FileStream::e_aUserRead ) )
   {
      setError( "Serialization to file failed: " + sFile );
      return false;
   }

   return m_dataLock.item().serialize( &fs ) == Item::sc_ok;
}

bool FileSessionData::dispose()
{
   String sFile = m_tmpDir + "/" + sID() + FALCON_SESSION_DATA_EXT;
   int32 fsStatus;

   return Sys::fal_unlink( sFile, fsStatus );
}

//============================================================
// File based session manager
//

FileSessionManager::FileSessionManager( const String& tmpDir ):
      m_tmpDir( tmpDir )
{

}


void FileSessionManager::startup()
{
   // delete all the expired sessions.
   if( timeout() == 0 )
      return;

   int32 err;
   DirEntry* entry = Sys::fal_openDir( m_tmpDir, err );
   if( entry == 0 )
      return;

   String name;
   TimeStamp now;
   now.currentTime();

   while( entry->read( name ) )
   {
      if ( name.endsWith( FALCON_SESSION_DATA_EXT ) )
      {
         String sFullName = m_tmpDir + "/" + name;

         FileStat st;
         if( Sys::fal_stats( sFullName, st ) )
         {
            st.m_mtime->distance( now );
            if( (st.m_mtime->m_day * 3600*24 +
                  st.m_mtime->m_hour * 3600 +
                  st.m_mtime->m_minute * 60 +
                  st.m_mtime->m_second ) > (int) timeout() )
            {
               Sys::fal_unlink( sFullName, err );
            }
         }
      }
   }
}

FileSessionManager::~FileSessionManager()
{
}

void FileSessionManager::configFromModule( const Module* mod )
{
   AttribMap* attribs = mod->attributes();
   if( attribs == 0 )
   {
      return;
   }

   VarDef* value = attribs->findAttrib( FALCON_WOPI_TEMPDIR_ATTRIB );
   if( value != 0 && value->isString() )
   {
      m_tmpDir.bufferize( *value->asString() );
   }

   SessionManager::configFromModule( mod );
}

SessionData* FileSessionManager::createSession( const Falcon::String& sSID )
{
   return new FileSessionData( sSID, m_tmpDir );
}

}
}

/* end of file_sm.cpp */
