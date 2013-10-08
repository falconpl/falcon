/*
   FALCON - The Falcon Programming Language.
   FILE: options.h

   Web Oriented Programming Interface

   Object encapsulating requests.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 23 Apr 2010 11:24:16 -0700

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef WOPI_OPTIONS_EXT_H
#define WOPI_OPTIONS_EXT_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>
#include <falcon/class.h>

namespace Falcon {
namespace WOPI {

class Options
{
public:
   Options();
   virtual ~Options();

   virtual const String& getTempPath() const { return m_sTempPath; }
   virtual int64 getMemoryUpload() const { return m_nMaxMemUpload; }

   //! Sets the maximum size that for uploading a part.
   void setMaxMemUpload( int64 mm ) { m_nMaxMemUpload = mm; }

   //! Sets the location for temporary files.
   void setUploadPath( const String& path ) { m_sTempPath = path; }
private:
   String m_sTempPath;
   int64 m_nMaxMemUpload;
};


class ClassOptions: public Class
{
public:

};

}}

#endif

/* end of options.h */
