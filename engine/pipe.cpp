/*
   FALCON - The Falcon Programming Language.
   FILE: pipe.cpp

   System independent abstraction for linked inter-process sockets.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 22 Feb 2013 20:01:34 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/pipe.cpp"

#include <falcon/pipe.h>
#include <falcon/stderrors.h>
#include <falcon/filedatampx.h>
#include <falcon/selector.h>
#include <falcon/pipestreams.h>

namespace Falcon {
namespace Sys {

Pipe::~Pipe()
{
   close();
   delete m_readSide;
   delete m_writeSide;
}

void Pipe::close()
{
   closeRead();
   closeWrite();
}


Stream* Pipe::getReadStream()
{
   if( m_readSide != 0 )
   {
      FileData* fd = m_readSide;
      m_readSide = 0;
      return new ReadPipeStream( fd );
   }
   return 0;
}


Stream* Pipe::getWriteStream()
{
   if( m_writeSide != 0 )
   {
      FileData* fd = m_writeSide;
      m_writeSide = 0;
      return new WritePipeStream( fd );
   }
   return 0;
}

//======================================================
// Pipe Traits
//======================================================

class Pipe::MpxFactory::ReadMPX: public Sys::FileDataMPX
{
public:
   ReadMPX( const Multiplex::Factory* generator, Selector* master ):
      FileDataMPX( generator, master )
   {}

   virtual ~ReadMPX() {}

   virtual void add( Selectable* stream, int mode )
   {
      if( (mode & Selector::mode_read) != 0 )
      {
         FileDataMPX::add(stream, Selector::mode_read);
      }
   }
};

class Pipe::MpxFactory::WriteMPX: public Sys::FileDataMPX
{
public:
   WriteMPX( const Multiplex::Factory* generator, Selector* master ):
      FileDataMPX( generator, master )
   {}

   virtual ~WriteMPX() {}

   // This is system specific
   virtual void add( Selectable* stream, int mode )
   {
      if( (mode & Selector::mode_write) != 0 )
      {
         FileDataMPX::add(stream, Selector::mode_write);
      }
   }
};


Pipe::MpxFactory::MpxFactory( bool readDirection ):
      m_bReadDirection(readDirection)
{}

Pipe::MpxFactory::~MpxFactory()
{}

Multiplex* Pipe::MpxFactory::create( Selector* master ) const
{
   if( m_bReadDirection )
   {
      return new ReadMPX( this, master );
   }
   else {
      return new WriteMPX(this, master );
   }
}

}
}

/* end of pipe.cpp */
