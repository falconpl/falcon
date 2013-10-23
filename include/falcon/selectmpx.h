/*
   FALCON - The Falcon Programming Language.
   FILE: selectmpx.h

   Multiplexer for blocking system-level streams -- using SELECT
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 02 Mar 2013 11:20:54 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SYS_SELECTMPX_H_
#define _FALCON_SYS_SELECTMPX_H_

#include <falcon/setup.h>
#include <falcon/multiplex.h>

namespace Falcon {
namespace Sys {


/**
   Multiplexer for blocking system-level streams -- using SELECT

   This multicplexer is used on those systems and for those 
   circumstances where a system-level file descriptor is available,
   and the select() interface is available as well, but it's not
   possible to use the FDataMPX based on POSIX poll() call.

   This includes, among other things, the MS-Windows version of the
   socket selector and OSX version of the file data selector.

   The MPX selector supposes having a FD-Selectable resource 
   (casts its selectable into a fd-selectable) and then uses the
   fd member to get a system-level file descriptor to be fed into
   select().
*/
class SelectMPX: public Multiplex
{
public:
   SelectMPX( const Multiplex::Factory* generator, Selector* master );
   virtual ~SelectMPX();

   virtual void add( Selectable* stream, int mode );
   virtual void remove( Selectable* stream );
   virtual uint32 size() const;

   #ifndef __MINGW32__
   typedef int FILE_DESCRIPTOR;
   #else
   typedef unsigned int FILE_DESCRIPTOR;
   #endif

   virtual int readControlFD( void* data, int size ) const = 0;
   virtual int writeControlFD( const void* data, int size ) const = 0;
   virtual FILE_DESCRIPTOR getSelectableControlFD() const = 0;

protected:
   /* Call this from subclass destructor prior destroying the controller FD!! */
   void quit();

private:
   class Private;
   Private* _p;
   friend class Private;

   uint32 m_size;   
};

}
}

#endif /* _FALCON_SYS_SELECT_H_ */

/* end of selectmpx.h */
