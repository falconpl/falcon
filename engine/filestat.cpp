/*
   FALCON - The Falcon Programming Language
   FILE: filestats.cpp
   $Id: filestat.cpp,v 1.3 2007/07/06 08:14:38 jonnymind Exp $

    Directory and file specific statistic accounting
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: gio giu 21 2007
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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

bool FileStat::isReflective() const
{
   return true;
}

void FileStat::getProperty( const String &propName, Item &prop )
{
   if( propName == "type" )
      prop = (int64) m_type;
   else if( propName == "size" )
      prop = m_size;
   else if( propName == "owner" )
      prop = (int64) m_owner;
   else if( propName == "group" )
      prop = (int64) m_group;
   else if( propName == "access" )
      prop = (int64) m_access;
   else if( propName == "attribs" )
      prop = (int64) m_attribs;
}

void FileStat::setProperty( const String &propName, Item &prop )
{
  if( propName == "type" )
      m_type = (e_fileType) prop.asInteger();
   else if( propName == "size" )
      m_size = prop.asInteger();
   else if( propName == "owner" )
      m_owner = (uint32) prop.asInteger();
   else if( propName == "group" )
      m_group = (uint32) prop.asInteger();
   else if( propName == "access" )
      m_access = (uint32) prop.asInteger();
   else if( propName == "attribs" )
      m_attribs = (uint32) prop.asInteger();
}

UserData *FileStat::clone() const
{
   FileStat *other = new FileStat( *this );
   return other;
}

}


/* end of filestat.cpp */
