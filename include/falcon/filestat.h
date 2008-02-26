/*
   FALCON - The Falcon Programming Language.
   FILE: filestat.h
   $Id: filestat.h,v 1.4 2007/06/26 20:40:03 jonnymind Exp $

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

#ifndef flc_filestat_H
#define flc_filestat_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/userdata.h>

namespace Falcon {

class String;
class TimeStamp;

/** Multiplatform statistics on files. */
class FALCON_DYN_CLASS FileStat: public UserData
{
public:
   typedef enum {
      t_notFound = -1,
      t_unknown = 0,
      t_normal = 1,
      t_dir = 2,
      t_pipe = 3,
      t_link = 4,
      t_device = 5,
      t_socket = 6,
   } e_fileType;

   e_fileType m_type;
   int64 m_size;
   int32 m_owner;
   int32 m_group;
   int32 m_access;
   /** Dos attribs */
   int32 m_attribs;
   /** Creation or status change */
   TimeStamp *m_ctime;
   /** Last write time */
   TimeStamp *m_mtime;
   /** Last access time */
   TimeStamp *m_atime;

   FileStat();
   FileStat( const FileStat &other );
   virtual ~FileStat();

   virtual bool isReflective() const;
   virtual void setProperty( const String &propName, Item &prop );
   virtual void getProperty( const String &propName, Item &prop );
   virtual UserData * clone() const;
};

}

#endif

/* end of filestat.h */
