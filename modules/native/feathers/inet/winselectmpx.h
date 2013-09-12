/*
   FALCON - The Falcon Programming Language.
   FILE: winselectmpx.h

   Multiplex using select() Winsock2 call.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 08 Sep 2013 13:47:28 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_WINSELECT_MPX_H_
#define _FALCON_WINSELECT_MPX_H_

#include <falcon/setup.h>
#include <falcon/selectmpx.h>

namespace Falcon {
namespace Mod {

class WinSelectMPXFactory: public Multiplex::Factory
{
public:
   WinSelectMPXFactory();
   virtual ~WinSelectMPXFactory();
   virtual Multiplex* create( Selector* selector ) const;
};


class WinSelectMPX: public Sys::SelectMPX
{
public:
   WinSelectMPX( const Multiplex::Factory* fact, Selector* master );
   virtual ~WinSelectMPX();

   virtual int readControlFD( void* data, int size ) const;
   virtual int writeControlFD( const void* data, int size ) const;
   virtual FILE_DESCRIPTOR getSelectableControlFD() const;

   
private:
   FILE_DESCRIPTOR m_ctrlSend;
   FILE_DESCRIPTOR m_ctrlRecv;
   bool makeCtrlFD();
};

}
}

#endif

/* end of winselectmpx.cpp */
