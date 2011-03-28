/*
   FALCON - The Falcon Programming Language.
   FILE: vfsiface.cpp

   Generic provider of file system abstraction.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 28 Mar 2011 17:26:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/vfsiface.h>
#include <falcon/ioerror.h>

namespace Falcon {

VFSIface::VFSIface():
   VFSProvider("")
{
}

VFSIface::~VFSIface()
{
   VFSMap::iterator iter = m_vfsmap.begin();
   while( iter != m_vfsmap.end() )
   {
      delete iter->second;
      ++iter;
   }
}

Stream* VFSIface::open( const URI &uri, const OParams &p )
{
   VFSMap::iterator iter = m_vfsmap.find( uri.scheme() );
   if( iter == m_vfsmap.end() )
   {
      throw new IOError( ErrorParam(e_io_unsup, __LINE__, __FILE__)
              .extra("scheme " + uri.scheme()) );
   }

   return iter->second->open( uri, p );
}

Stream* VFSIface::create( const URI &uri, const CParams &p )
{
   VFSMap::iterator iter = m_vfsmap.find( uri.scheme() );
   if( iter == m_vfsmap.end() )
   {
      throw new IOError( ErrorParam(e_io_unsup, __LINE__, __FILE__)
              .extra("scheme " + uri.scheme()) );
   }

   return iter->second->create( uri, p );
}

Directory* VFSIface::openDir( const URI &uri )
{
   VFSMap::iterator iter = m_vfsmap.find( uri.scheme() );
   if( iter == m_vfsmap.end() )
   {
      throw new IOError( ErrorParam(e_io_unsup, __LINE__, __FILE__)
              .extra("scheme " + uri.scheme()) );
   }

   return iter->second->openDir( uri );
}


bool VFSIface::readStats( const URI &uri, FileStat &s )
{
   VFSMap::iterator iter = m_vfsmap.find( uri.scheme() );
   if( iter == m_vfsmap.end() )
   {
      throw new IOError( ErrorParam(e_io_unsup, __LINE__, __FILE__)
              .extra("scheme " + uri.scheme()) );
   }

   return iter->second->readStats( uri, s );
}


FileStat::t_fileType VFSIface::fileType( const URI& uri )
{
   VFSMap::iterator iter = m_vfsmap.find( uri.scheme() );
   if( iter == m_vfsmap.end() )
   {
      throw new IOError( ErrorParam(e_io_unsup, __LINE__, __FILE__)
              .extra("scheme " + uri.scheme()) );
   }

   return iter->second->fileType( uri );
}


void VFSIface::mkdir( const URI &uri, bool bCreateParent )
{
   VFSMap::iterator iter = m_vfsmap.find( uri.scheme() );
   if( iter == m_vfsmap.end() )
   {
      throw new IOError( ErrorParam(e_io_unsup, __LINE__, __FILE__)
              .extra("scheme " + uri.scheme()) );
   }

   return iter->second->mkdir( uri, bCreateParent );
}

void VFSIface::erase( const URI &uri )
{
   VFSMap::iterator iter = m_vfsmap.find( uri.scheme() );
   if( iter == m_vfsmap.end() )
   {
      throw new IOError( ErrorParam(e_io_unsup, __LINE__, __FILE__)
              .extra("scheme " + uri.scheme()) );
   }

   return iter->second->erase( uri );
}


void VFSIface::move( const URI &suri, const URI &duri )
{
   VFSMap::iterator iter = m_vfsmap.find( suri.scheme() );

   if( iter == m_vfsmap.end() )
   {
      throw new IOError( ErrorParam(e_io_unsup, __LINE__, __FILE__)
              .extra("scheme " + suri.scheme()) );
   }

   return iter->second->move( suri, duri );
}


void VFSIface::addVFS( const String& str, VFSProvider* vfs )
{
   VFSMap::iterator iter = m_vfsmap.find( str );
   if( iter != m_vfsmap.end() )
   {
      delete iter->second;
   }
   m_vfsmap[str] = vfs;
}


VFSProvider* VFSIface::getVFS( const String& str )
{
   VFSMap::iterator iter = m_vfsmap.find( str );
   if( iter != m_vfsmap.end() )
   {
      return iter->second;
   }
   
   return 0;
}

}

/* end of vfsiface.cpp */
