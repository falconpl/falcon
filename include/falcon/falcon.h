/*
   FALCON - The Falcon Programming Language.
   FILE: falcon.h

   Main inclusion file file
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 25 Jul 2011 15:17:33 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_FALCON_H
#define	FALCON_FALCON_H

// This includes the vast majority of things in Falcon.
#include <falcon/engine.h>
#include <falcon/log.h>

#include <falcon/string.h>
#include <falcon/autocstring.h>
#include <falcon/autowstring.h>

#include <falcon/expression.h>
#include <falcon/statement.h>
#include <falcon/psteps/exprrule.h>

#include <falcon/stdstreams.h>
#include <falcon/streambuffer.h>
#include <falcon/stringstream.h>
#include <falcon/textwriter.h>
#include <falcon/textreader.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>

#include <falcon/debugger.h>


#include <falcon/module.h>
#include <falcon/intcompiler.h>
#include <falcon/modcompiler.h>
#include <falcon/modspace.h>
#include <falcon/modloader.h>
#include <falcon/process.h>
#include <falcon/loaderprocess.h>

#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/wvmcontext.h>
#include <falcon/qvmcontext.h>
#include <falcon/application.h>

#include <falcon/cm/coremodule.h>
#include <falcon/vfsprovider.h>

#include <falcon/symbol.h>

#include <falcon/item.h>
#include <falcon/itemid.h>

#include <falcon/itemdict.h>
#include <falcon/itemarray.h>

#include <falcon/gclock.h>

//--- error headers ---
#include <falcon/classes/classerror.h>
#include <falcon/stderrors.h>


#endif	/* FALCON_H */

/* end of falcon.h */
