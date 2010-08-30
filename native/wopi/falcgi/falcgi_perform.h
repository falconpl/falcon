/*
   FALCON - The Falcon Programming Language.
   FILE: cgi_perform.h

   Falcon CGI program driver - part running a single script.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 20 Feb 2010 17:35:14 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef CGI_PERFORM_H_
#define CGI_PERFORM_H_

void* perform( CGIOptions& options, int argc, char* argv[] );

#endif /* CGI_PERFORM_H_ */

/* end of cgi_perform.h */
