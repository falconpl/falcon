#ifndef GTK_BUILDER_HPP
#define GTK_BUILDER_HPP

#include "modgtk.hpp"

namespace Falcon {
namespace Gtk {

class Builder: public CoreGObject
{
public:
    Builder( const Falcon::CoreClass*, const GtkBuilder* );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC add_from_file( VMARG );

    static FALCON_FUNC get_object( VMARG ); 
};

}
}

#endif // GTK_BUILDER_HPP
