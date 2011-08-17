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

class VFSIface_p 
{
public:
   typedef std::map<String, VFSProvider* > VFSMap;
   VFSMap m_vfsmap;
};


VFSIface::VFSIface():
   VFSProvider("")
{
   _p = new VFSIface_p;
}

VFSIface::~VFSIface()
{
   VFSIface_p::VFSMap::iterator iter = _p->m_vfsmap.begin();
   while( iter != _p->m_vfsmap.end() )
   {
      delete iter->second;
      ++iter;
   }

   delete _p;
}

Stream* VFSIface::open( const URI &uri, const OParams &p )
{
   VFSIface_p::VFSMap::iterator iter = _p->m_vfsmap.find( uri.scheme() );
   if( iter == _p->m_vfsmap.end() )
   {
      throw new IOError( ErrorParam(e_io_unsup, __LINE__, __FILE__)
              .extra("scheme " + uri.scheme()) );
   }

   return iter->second->open( uri, p );
}

Stream* VFSIface::create( const URI &uri, const CParams &p )
{
   VFSIface_p::VFSMap::iterator iter = _p->m_vfsmap.find( uri.scheme() );
   if( iter == _p->m_vfsmap.end() )
   {
      throw new IOError( ErrorParam(e_io_unsup, __LINE__, __FILE__)
              .extra("scheme " + uri.scheme()) );
   }

   return iter->second->create( uri, p );
}

Directory* VFSIface::openDir( const URI &uri )
{
   VFSIface_p::VFSMap::iterator iter = _p->m_vfsmap.find( uri.scheme() );
   if( iter == _p->m_vfsmap.end() )
   {
      throw new IOError( ErrorParam(e_io_unsup, __LINE__, __FILE__)
              .extra("scheme " + uri.scheme()) );
   }

   return iter->second->openDir( uri );
}


bool VFSIface::readStats( const URI &uri, FileStat &s, bool deref )
{
   VFSIface_p::VFSMap::iterator iter = _p->m_vfsmap.find( uri.scheme() );
   if( iter == _p->m_vfsmap.end() )
   {
      throw new IOError( ErrorParam(e_io_unsup, __LINE__, __FILE__)
              .extra("scheme " + uri.scheme()) );
   }

   return iter->second->readStats( uri, s, deref );
}


FileStat::t_fileType VFSIface::fileType( const URI& uri, bool bderef )
{
   VFSIface_p::VFSMap::iterator iter = _p->m_vfsmap.find( uri.scheme() );
   if( iter == _p->m_vfsmap.end() )
   {
      throw new IOError( ErrorParam(e_io_unsup, __LINE__, __FILE__)
              .extra("scheme " + uri.scheme()) );
   }

   return iter->second->fileType( uri, bderef );
}


void VFSIface::mkdir( const URI &uri, bool bCreateParent )
{
   VFSIface_p::VFSMap::iterator iter = _p->m_vfsmap.find( uri.scheme() );
   if( iter == _p->m_vfsmap.end() )
   {
      throw new IOError( ErrorParam(e_io_unsup, __LINE__, __FILE__)
              .extra("scheme " + uri.scheme()) );
   }

   return iter->second->mkdir( uri, bCreateParent );
}

void VFSIface::erase( const URI &uri )
{
   VFSIface_p::VFSMap::iterator iter = _p->m_vfsmap.find( uri.scheme() );
   if( iter == _p->m_vfsmap.end() )
   {
      throw new IOError( ErrorParam(e_io_unsup, __LINE__, __FILE__)
              .extra("scheme " + uri.scheme()) );
   }

   return iter->second->erase( uri );
}


void VFSIface::move( const URI &suri, const URI &duri )
{
   VFSIface_p::VFSMap::iterator iter = _p->m_vfsmap.find( suri.scheme() );

   if( iter == _p->m_vfsmap.end() )
   {
      throw new IOError( ErrorParam(e_io_unsup, __LINE__, __FILE__)
              .extra("scheme " + suri.scheme()) );
   }

   return iter->second->move( suri, duri );
}


void VFSIface::addVFS( const String& str, VFSProvider* vfs )
{
   VFSIface_p::VFSMap::iterator iter = _p->m_vfsmap.find( str );
   if( iter != _p->m_vfsmap.end() )
   {
      delete iter->second;
   }
   _p->m_vfsmap[str] = vfs;
}


VFSProvider* VFSIface::getVFS( const String& str )
{
   VFSIface_p::VFSMap::iterator iter = _p->m_vfsmap.find( str );
   if( iter != _p->m_vfsmap.end() )
   {
      return iter->second;
   }
   
   return 0;
}

}

/* end of vfsiface.cpp */
