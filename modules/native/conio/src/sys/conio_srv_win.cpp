/*
   FALCON - The Falcon Programming Language.
   FILE: conio_srv_win.cpp

   Basic Console I/O support
   Interface extension functions
   -------------------------------------------------------------------
   Author: Unknown author
   Begin: Thu, 05 Sep 2008 20:12:14 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: The above AUTHOR

      Licensed under the Falcon Programming Language License,
   Version 1.1 (the "License"); you may not use this file
   except in compliance with the License. You may obtain
   a copy of the License at

      http://www.falconpl.org/?page_id=license_1_1

   Unless required by applicable law or agreed to in writing,
   software distributed under the License is distributed on
   an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied. See the License for the
   specific language governing permissions and limitations
   under the License.

*/

#include <falcon/fassert.h>
#include "conio_srv_win.h"

namespace Falcon {
namespace Srv {

ConsoleSrv::ConsoleSrv():
   Service( "CONSOLE" ),
   m_sys(0)
{
}

// be sure to shutdown:
ConsoleSrv::~ConsoleSrv()
{
   shutdown();
}

ConsoleSrv::error_type ConsoleSrv::init()
{
   // already initialized
   if ( m_sys != 0 )
      return e_dbl_init;
      
   // we must create our system-specific data
   // try to alloc a console; the result doesn't really matters
   AllocConsole();
   
   // now, if opening the console fails THIS matters.
   HANDLE hIn = GetStdHandle(STD_INPUT_HANDLE);
   HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
   m_sys = new ConioSrvSys( hIn, hOut );
   // we know we have in and out streams after an AllocConsole
   // but what about the buffer info?
   if( !GetConsoleScreenBufferInfo( m_sys->hConsoleOut, &m_sys->csbi ) )
      return e_init;
   
   //todo; setup input/output masks, CTRL- handlers etc.
   
   // Great, we're on business; emulate a CLS to match other behaviors
   cls();
   return e_none;
}


ConsoleSrv::error_type ConsoleSrv::cls()
{
   if ( m_sys == 0 )
      return e_not_init;
      
   COORD coordScreen = { 0, 0 };    // home for the cursor 
   DWORD cCharsWritten;
   DWORD dwConSize;

   // Get the number of character cells in the current buffer. 
   dwConSize = m_sys->csbi.dwSize.X * m_sys->csbi.dwSize.Y;

   // Fill the entire screen with blanks.
   if( !FillConsoleOutputCharacter(  m_sys->hConsoleOut, (TCHAR) ' ',
      dwConSize, coordScreen, &cCharsWritten ))
      return e_write;

   // Get the current text attribute.
   if( !GetConsoleScreenBufferInfo( m_sys->hConsoleOut, &m_sys->csbi ))
      return e_write;

   // Set the buffer's attributes accordingly.

   if( !FillConsoleOutputAttribute( m_sys->hConsoleOut, m_sys->csbi.wAttributes,
      dwConSize, coordScreen, &cCharsWritten ))
      return e_write;

   // Put the cursor at its home coordinates.
   SetConsoleCursorPosition( m_sys->hConsoleOut, coordScreen );
   return e_none;
}

   
void ConsoleSrv::shutdown()
{
   cls();
   delete m_sys;
   m_sys = 0;
}

}
}

/* end of conio_srv_win.cpp */
