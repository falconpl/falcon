/*
   FALCON - The Falcon Programming Language.
   FILE: filestat.h

   Directory and file specific statistic accounting
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 25 Mar 2011 18:21:38 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Directory and file specific statistic accounting
*/

#ifndef _FALCON_FILESTAT_H_
#define _FALCON_FILESTAT_H_

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/timestamp.h>

namespace Falcon {

class String;

/** Multiplatform statistics on files. */
class FALCON_DYN_CLASS FileStat
{
public:
   typedef enum {
      _notFound = -1,
      _unknown = 0,
      _normal = 1,
      _dir = 2,
      _pipe = 3,
      _link = 4,
      _device = 5,
      _socket = 6
   } t_fileType;

   FileStat();
   FileStat( const FileStat &other );
   virtual ~FileStat();

   bool exists() const { return m_type != _notFound; }
   bool isFile() const { return m_type == _normal; }
   bool isDirectory() const { return m_type == _dir; }

   /** Type of the found file. */
   t_fileType type() const { return m_type; }
   void type(t_fileType t) { m_type = t; }

   /** Size of the required file */
   int64 size() const { return m_size; }

   void size( int64 t ) { m_size = t; }

   /** Creation or status change */
   const TimeStamp& ctime() const { return m_ctime; }
   TimeStamp& ctime() { return m_ctime; }

   /** Last write time */
   const TimeStamp& mtime() const { return m_mtime; }
   TimeStamp& mtime() { return m_mtime; }

   /** Last access time */
   const TimeStamp& atime() const { return m_atime; }
   TimeStamp& atime() { return m_atime; }

private:
   /** Type of the found file. */
   t_fileType m_type;

   /** Size of the required file */
   int64 m_size;

   /** Creation or status change */
   TimeStamp m_ctime;

   /** Last write time */
   TimeStamp m_mtime;

   /** Last access time */
   TimeStamp m_atime;
};

}

#endif

/* end of filestat.h */
