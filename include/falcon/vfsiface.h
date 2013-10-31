/*
   FALCON - The Falcon Programming Language.
   FILE: vfsiface.h

   Generic provider of file system abstraction.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 28 Mar 2011 17:26:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Generic provider of file system abstraction.
*/

#ifndef FALCON_VFSIFACE_H
#define FALCON_VFSIFACE_H

#include <falcon/vfsprovider.h>

#include <map>

namespace Falcon {

class VFSIface_p;

/** Interface to the Virtual File Systems.

 This class interpreets the requests of the querier selecting the appropriate
 virtual file system provider.

 All the virtual file systems register to a singleton instance of this class
 that is allocatated in the engine and can be accessed through the Engine::vsf()
 accessor.

 If the protocol declared in the URI is not served by any provided
*/
class FALCON_DYN_CLASS VFSIface: public VFSProvider
{
public:

   VFSIface();
   virtual ~VFSIface();

   virtual Stream* open( const URI &uri, const OParams &p );
   virtual Stream* create( const URI &uri, const CParams &p);
   virtual Directory* openDir( const URI &uri );
   virtual bool readStats( const URI &uri, FileStat &s, bool deref );
   virtual FileStat::t_fileType fileType( const URI& uri, bool deref );

   virtual void mkdir( const URI &uri, bool bCreateParent=true );
   virtual void erase( const URI &uri );
   virtual void move( const URI &suri, const URI &duri );

   virtual void setCWD( const URI &uri );
   virtual void getCWD( URI &uri );

   void addVFS( const String& str, VFSProvider* vfs );
   VFSProvider* getVFS( const String& str );

private:
   VFSIface_p* _p;
};

}

#endif

/* end of vfsiface.h */
