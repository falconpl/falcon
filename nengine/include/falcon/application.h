/*
   FALCON - The Falcon Programming Language.
   FILE: application.h

   Utility to create embedding applications.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Feb 2011 13:14:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_APPLICATION_H
#define	FALCON_APPLICATION_H

#include <falcon/setup.h>

namespace Falcon {

/** Base class for embedding applications.

 This utility class takes care of properly initialize and shutdown the Falcon
 engine, so that it can be used to create safe embeddings.

 While calling Engine::init() and Engine::shutdown is mandatory, using this
 class is not necessary, but it may be a good practice for embeddings to store
 all the application-wise data that must be shared with Falcon inside a derived
 class.

 This class checks if it has been intantiated just once; if not, it terminates
 the program with an exit(1).
 */
class FALCON_DYN_CLASS Application
{
public:
   Application();
   ~Application();

private:
   static bool m_bUnique;
};

}

#endif	/* FALCON_ERROR_H */

/* end of application.h */
