/*
   FALCON - The Falcon Programming Language.
   FILE: filedatampx.h

   Multiplexer for blocking system-level streams.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 02 Mar 2013 11:20:54 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SYS_FILEDATAMPX_H_
#define _FALCON_SYS_FILEDATAMPX_H_

#include <falcon/setup.h>
#include <falcon/multiplex.h>

namespace Falcon {
namespace Sys {

/** Multiplexer for blocking system-level streams.
*
* This traits can be used for all those streams that are based on an underlying
* system FileData incarnation which refers to a handle or file descriptor open
* towards a blocking resource, which can be waited upon through standard system
* means.
*
* On POSIX this includes:
* - UNIX sockets.
* - TCP sockets.
* - Pipes (anonymous, named).
* - Standard streams (stdin/stdout/stderr).
* - Local files having a file descriptor (although they will be normally processed by DiskMPX)
* - Network files.
*
* On MS-Windows:
* - Pipes (anonymous, named)
* - Standard streams.
*
* \note In the POSIX implementation Multiplexed entities are considered to be
* FDSelector subclasses. Although it's possible to use a FDSubclass on MS-Windows
* as well, the multiplex won't make this assumption and just use the Selector interface.
*
*/
class FALCON_DYN_CLASS FileDataMPX: public Multiplex
{
public:
   FileDataMPX( const Multiplex::Factory* generator, Selector* master );
   virtual ~FileDataMPX();

   virtual void add( Selectable* stream, int mode );
   virtual void remove( Selectable* stream );
   virtual uint32 size() const;

private:
   class Private;
   Private* _p;

   uint32 m_size;
};

/**
 * Factory for the FileData multiplexer.
 */
class FALCON_DYN_CLASS FileDataMPXFactory: public Multiplex::Factory
{
public:
   FileDataMPXFactory() {}
   virtual ~FileDataMPXFactory() {}
   virtual Multiplex* create( Selector* selector ) const
   {
      return new FileDataMPX(this, selector);
   }
};

}
}

#endif /* _FALCON_SYS_FILEDATAMPX_H_ */

/* end of filedatampx.h */
