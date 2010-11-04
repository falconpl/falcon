#include <falcon/setup.h>
#include <falcon/types.h>

#ifndef PGSQL_EXT_H
#define PGSQL_EXT_H

namespace Falcon {

class VMachine;

namespace Ext {

FALCON_FUNC PgSQL_init( VMachine* vm );
FALCON_FUNC PgSQL_prepareNamed( VMachine* vm );

} // !Ext
} // !Falcon

#endif /* PGSQL_EXT_H */
