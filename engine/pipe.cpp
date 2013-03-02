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
#include <falcon/errors/ioerror.h>
#include <falcon/fstream.h>
#include <falcon/filedatampx.h>
#include <falcon/selector.h>

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


ReadOnlyFStream* Pipe::getReadStream()
{
   if( m_readSide != 0 )
   {
      FileData* fd = m_readSide;
      m_readSide = 0;
      return new ReadOnlyFStream( fd );
   }
   return 0;
}


WriteOnlyFStream* Pipe::getWriteStream()
{
   if( m_writeSide != 0 )
   {
      FileData* fd = m_writeSide;
      m_writeSide = 0;
      return new WriteOnlyFStream( fd );
   }
   return 0;
}

//======================================================
// Pipe Traits
//======================================================

class Pipe::Traits::ReadMPX: public Sys::FileDataMPX
{
public:
   ReadMPX( const StreamTraits* generator, Selector* master ):
      FileDataMPX( generator, master )
   {}

   virtual ~ReadMPX() {}

   virtual void addStream( Stream* stream, int mode )
   {
      if( (mode & Selector::mode_read) != 0 )
      {
         FileDataMPX::addStream(stream, Selector::mode_read);
      }
   }
};

class Pipe::Traits::WriteMPX: public Sys::FileDataMPX
{
public:
   WriteMPX( const StreamTraits* generator, Selector* master ):
      FileDataMPX( generator, master )
   {}

   virtual ~WriteMPX() {}

   // This is system specific
   virtual void addStream( Stream* stream, int mode )
   {
      if( (mode & Selector::mode_write) != 0 )
      {
         FileDataMPX::addStream(stream, Selector::mode_write);
      }
   }
};


Pipe::Traits::Traits( bool readDirection ):
      StreamTraits("Pipe::Traits", 0),
      m_bReadDirection(readDirection)
{}

Pipe::Traits::~Traits()
{}

Multiplex* Pipe::Traits::multiplex( Selector* master ) const
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
