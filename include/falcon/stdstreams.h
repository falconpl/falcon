/*
   FALCON - The Falcon Programming Language.
   FILE: vm_stdstreams.h

   System dependant default I/O streams.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven ago 25 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   System dependant default I/O streams.
   This file contains the declaration of the standard streams that will fill
   VM and other systems basic I/O systems and that may be system dependant.
*/

#ifndef flc_stdstreams_H
#define flc_stdstreams_H


namespace Falcon {

class Stream;

   /** System specific input stream factory function.
      This function will return a text oriented input stream.
   */
   FALCON_DYN_SYM Stream *stdInputStream();
   /** System specific output stream factory function. */
   FALCON_DYN_SYM Stream *stdOutputStream();
   /** System specific error stream factory function. */
   FALCON_DYN_SYM Stream *stdErrorStream();

   /** Default text converter.
      Depending on the system, this method will wrap the underlying stream in a EOL transcoder
      or it will just return it.
   */
   FALCON_DYN_SYM Stream *DefaultTextTranscoder( Stream *underlying, bool own = true );

   /** Adds just the sytem default EOL transcoding policy. */
   FALCON_DYN_SYM Stream *AddSystemEOL( Stream *underlying, bool own = true );

}

#endif

/* end of stdstreams.h */
