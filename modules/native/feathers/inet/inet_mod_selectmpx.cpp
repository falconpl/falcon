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


#include <falcon/diskmpxfactory.h>
#include <falcon/multiplex.h>
#include <falcon/selector.h>
#include <falcon/stream.h>

namespace Falcon
{

class DiskMpxFactory::MPX: public Multiplex
{
public:
   MPX( const DiskMpxFactory* generator, Selector* master );
   virtual ~MPX();

   virtual void add( Selectable* stream, int mode );
   virtual void remove( Selectable* stream );
   virtual uint32 size() const { return m_size; }

private:
   uint32 m_size;
};


DiskMpxFactory::MPX::MPX( const DiskMpxFactory* generator, Selector* master ):
         Multiplex( generator, master ),
         m_size(0)
{
}

DiskMpxFactory::MPX::~MPX()
{
}


void DiskMpxFactory::MPX::add( Selectable* stream, int mode )
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

   m_size++;
}


void DiskMpxFactory::MPX::remove( Selectable* )
{
   // do nothing.
   m_size--;
}

//=================================================================
// Base DiskFileTraits
//=================================================================

Multiplex* DiskMpxFactory::create( Selector* master ) const
{
   return new MPX( this, master );
}

}

/* end of diskfiletraits.cpp */
