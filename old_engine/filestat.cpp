/*
   FALCON - The Falcon Programming Language
   FILE: filestats.cpp

    Directory and file specific statistic accounting
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: gio giu 21 2007

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
    Directory and file specific statistic accounting
*/

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/item.h>
#include <falcon/filestat.h>
#include <falcon/timestamp.h>

namespace Falcon {

FileStat::FileStat():
   m_type(t_notFound),
   m_size(0),
   m_owner(0),
   m_group(0),
   m_access(0),
   m_attribs(0),
   m_ctime( 0 ),
   m_mtime( 0 ),
   m_atime( 0 )
{}

FileStat::FileStat( const FileStat &other ):
   m_type( other.m_type ),
   m_size( other.m_size ),
   m_owner( other.m_owner ),
   m_group( other.m_group ),
   m_access( other.m_access ),
   m_attribs( other.m_attribs )
{
   if( other.m_ctime != 0 )
      m_ctime = (TimeStamp *) other.m_ctime->clone();
   else
      m_ctime = 0;

   if( other.m_atime != 0 )
      m_atime = (TimeStamp *) other.m_atime->clone();
   else
      m_atime = 0;

   if( other.m_mtime != 0 )
      m_mtime = (TimeStamp *) other.m_mtime->clone();
   else
      m_mtime = 0;
}

FileStat::~FileStat() {
   delete m_ctime;
   delete m_atime;
   delete m_mtime;
}

//===================================
// Reflection
//

FileStat *FileStat::clone() const
{
   FileStat *other = new FileStat( *this );
   return other;
}

}


/* end of filestat.cpp */
