/*
   FALCON - The Falcon Programming Language.
   FILE: diskfiletraits.cpp

   Traits for plain local disk files.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 02 Mar 2013 09:38:34 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/diskfiletraits.h>
#include <falcon/multiplex.h>
#include <falcon/selector.h>
#include <falcon/stream.h>

namespace Falcon
{

class DiskFileTraits::MPX: public Multiplex
{
public:
   MPX( const StreamTraits* generator, Selector* master );
   virtual ~MPX();

   virtual void addStream( Stream* stream, int mode );
   virtual void removeStream( Stream* stream );
};


DiskFileTraits::MPX::MPX( const StreamTraits* generator, Selector* master ):
         Multiplex( generator, master )
{
}

DiskFileTraits::MPX::~MPX()
{
}


void DiskFileTraits::MPX::addStream( Stream* stream, int mode )
{
   if( (mode & Selector::mode_write) != 0)
   {
      // always writable
      onReadyWrite(stream);
   }

   if( (mode & Selector::mode_read) != 0)
   {
      // always readable
      onReadyRead(stream);
   }

   // disk files never have extra/error/exceptional data.
   stream->decref();
}


void DiskFileTraits::MPX::removeStream( Stream* )
{
   // do nothing.
}

//=================================================================
// Base DiskFileTraits
//=================================================================

Multiplex* DiskFileTraits::multiplex( Selector* master ) const
{
   return new MPX( this, master );
}

}

/* end of diskfiletraits.cpp */
