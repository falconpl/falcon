/*
   FALCON - The Falcon Programming Language.
   FILE: winselectmpx.cpp

   Multiplex using select() Winsock2 call.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 08 Sep 2013 13:47:28 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "winselectmpx.h"
#include "inet_ext.h"

namespace Falcon {
namespace Mod {

//=================================================
// Factory
//=================================================

WinSelectMPXFactory::WinSelectMPXFactory()
{}
WinSelectMPXFactory::~WinSelectMPXFactory()
{}

Multiplex* WinSelectMPXFactory::create( Selector* selector ) const
{
   return new WinSelectMPX( this, selector );
}

//=================================================
// Multiplex
//=================================================


WinSelectMPX::WinSelectMPX( const Multiplex::Factory* fact, Selector* master ):
   SelectMPX(fact, master)
{
   if( ! makeCtrlFD() )
   {
      throw FALCON_SIGN_XERROR( Ext::NetError, FALSOCK_ERR_GENERIC,
         .desc( FALSOCK_ERR_GENERIC_MSG )
         .extra( "Creating multiplex socket pipe" )
         .sysError(WSAGetLastError()) );
   }
}

WinSelectMPX::~WinSelectMPX()
{
   if( m_ctrlSend != INVALID_SOCKET )
   {
      quit();
      ::closesocket(m_ctrlSend);
   }

   if( m_ctrlRecv != INVALID_SOCKET )
   {
      ::closesocket(m_ctrlRecv);
   }
}

int WinSelectMPX::readControlFD( void* data, int size ) const
{
   int res = ::recv( m_ctrlRecv, (char*) data, size, 0 );
   return res;
}

int WinSelectMPX::writeControlFD( const void* data, int size ) const
{
   int res = ::send( m_ctrlSend, (char*) data, size, 0 );
   return res;
}

WinSelectMPX::FILE_DESCRIPTOR WinSelectMPX::getSelectableControlFD() const
{
   return m_ctrlRecv;
}

bool WinSelectMPX::makeCtrlFD()
{
   SOCKET s;
 	struct sockaddr_in serv_addr;
 	int len = sizeof(serv_addr);

   m_ctrlRecv = m_ctrlSend = (FILE_DESCRIPTOR)INVALID_SOCKET;

 	if ((s = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET)
   {
 	   return false;
 	}

 	memset((void *) &serv_addr, 0, sizeof(serv_addr));
 	serv_addr.sin_family = AF_INET;
 	serv_addr.sin_port = htons(0);
 	serv_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

 	if (bind(s, (SOCKADDR *) & serv_addr, len) == SOCKET_ERROR)
 	{
      ::closesocket(s);
 	   return false;
 	}

   if (listen(s, 1) == SOCKET_ERROR)
 	{
      ::closesocket(s);
 	   return false;
 	}

   if (getsockname(s, (SOCKADDR *) & serv_addr, &len) == SOCKET_ERROR)
 	{
      ::closesocket(s);
 	   return false;
 	}

   if ((m_ctrlSend = socket(PF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET)
 	{
 	   closesocket(s);
 	   return false;
 	}

 	if (connect(m_ctrlSend, (SOCKADDR *) & serv_addr, len) == SOCKET_ERROR)
 	{
      ::closesocket(m_ctrlSend);
      m_ctrlSend = (FILE_DESCRIPTOR)INVALID_SOCKET;
      ::closesocket(s);
 	   return false;
 	}

   if ((m_ctrlRecv = accept(s, (SOCKADDR *) & serv_addr, &len)) == INVALID_SOCKET)
 	{
      ::closesocket(m_ctrlSend);
 	   m_ctrlSend = (FILE_DESCRIPTOR)INVALID_SOCKET;
      ::closesocket(s);
 	   return false;
 	}

   ::closesocket(s);
 	return true;
}


}
}

/* end of winselectmpx.cpp */
