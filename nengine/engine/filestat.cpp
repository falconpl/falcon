/*
   FALCON - The Falcon Programming Language.
   FILE: filestat.cpp

   Directory and file specific statistic accounting
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 25 Mar 2011 18:21:38 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/filestat.h>

namespace Falcon {

FileStat::FileStat():
   m_type(_notFound),
   m_size(0)
{
}


FileStat::FileStat( const FileStat &other )
{
}


FileStat::~FileStat()
{
}

}

/* end of filestat.cpp */

