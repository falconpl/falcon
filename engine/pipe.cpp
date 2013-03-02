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
}

void Pipe::close()
{
   closeRead();
   closeWrite();
}


ReadOnlyFStream* Pipe::getReadStream()
{
   FileData* fd = new FileData();
   m_readSide.passOn( *fd );
   return new ReadOnlyFStream( fd );
}


WriteOnlyFStream* Pipe::getWriteStream()
{
   FileData* fd = new FileData();
   m_writeSide.passOn( *fd );
   return new WriteOnlyFStream( fd );
}

//======================================================
// Pipe Traits
//======================================================

class Pipe::Traits::ReadMPX: public FileDataMPX
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

class Pipe::Traits::WriteMPX: public FileDataMPX
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
