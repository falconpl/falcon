/*
 * FALCON - The Falcon Programming Language.
 * FILE: oracle_ext.h
 *
 * Oracle Falcon extension interface
 * -------------------------------------------------------------------
 * Author: Steven Oliver
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include <falcon/module.h>

#ifndef ORACLE_EXT_H
#define ORACLE_EXT_H

namespace Falcon
{
    class VMachine;

    namespace Ext
    {
        FALCON_FUNC Oracle_init( VMachine *vm );
    }
}

#endif /* ORACLE_EXT_H */

/* end of oracle_ext.h */

