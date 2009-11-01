/*
   FALCON - The Falcon Programming Language.
   FILE: filestat.h

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

#ifndef flc_filestat_H
#define flc_filestat_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/falcondata.h>

namespace Falcon {

class String;
class TimeStamp;
class MemPool;

/** Multiplatform statistics on files. */
class FALCON_DYN_CLASS FileStat: public FalconData
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
      t_socket = 6
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

   virtual FileStat * clone() const;
   virtual void gcMark( uint32 mark ) {}
};

}

#endif

/* end of filestat.h */
