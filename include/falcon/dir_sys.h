/*
   FALCON - The Falcon Programming Language.
   FILE: dir_internal.h
   $Id: dir_sys.h,v 1.3 2007/06/23 10:14:51 jonnymind Exp $

   Internal functions prototypes for DirApi.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun feb 13 2006
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
   Internal functions prototypes for DirApi.

   This files holds the internal api for directories that is not to be published.
*/

#ifndef flc_dir_sys_H
#define flc_dir_sys_H

#include <falcon/filestat.h>
#include <falcon/userdata.h>

namespace Falcon {

/** Directory entry.

   This class encapsulate one directory entry, that is, one name found
   in directory searches.

   It has methods to read the next entry and to close the search.
*/
class DirEntry: public UserData
{

protected:
   uint32 m_lastError;

public:
   DirEntry():
      m_lastError(0)
   {}

   virtual bool read( String &fname ) = 0;
   virtual void close() = 0;
   uint32 lastError() const { return m_lastError; }
};

namespace Sys {

bool FALCON_DYN_SYM fal_fileType( const String &filename, FileStat::e_fileType &st );
bool FALCON_DYN_SYM fal_stats( const String &filename, FileStat &st );
bool FALCON_DYN_SYM fal_mkdir( const String &filename, int32 &fsStatus );
bool FALCON_DYN_SYM fal_unlink( const String &filename, int32 &fsStatus );
bool FALCON_DYN_SYM fal_rmdir( const String &filename, int32 &fsStatus );
bool FALCON_DYN_SYM fal_chdir( const String &filename, int32 &fsStatus );
bool FALCON_DYN_SYM fal_move( const String &filename, const String &dest, int32 &fsStatus );
bool FALCON_DYN_SYM fal_getcwd( String &fname, int32 &fsError );
bool FALCON_DYN_SYM fal_chmod( const String &fname, uint32 mode );
bool FALCON_DYN_SYM fal_chown( const String &fname, int32 owner );
bool FALCON_DYN_SYM fal_chgrp( const String &fname, int32 grp );
bool FALCON_DYN_SYM fal_readlink( const String &fname, String &link );
bool FALCON_DYN_SYM fal_writelink( const String &fname, const String &link );
DirEntry FALCON_DYN_SYM *fal_openDir( const String &path, int32 &fsError  );
void FALCON_DYN_SYM fal_closeDir( DirEntry *entry  );
}
}

#endif

/* end of dir_internal.h */
