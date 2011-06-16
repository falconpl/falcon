#ifndef GTK_SPINNER_HPP
#define GTK_SPINNER_HPP

#include "modgtk.hpp"

namespace Falcon {
namespace Gtk {

class Spinner: public Gtk::CoreGObject
{
public:
    Spinner( const CoreClass*, const GtkSpinner* );

    static CoreObject* factory( const CoreClass*, void*, bool );

    static void modInit( Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC start( VMARG );

    static FALCON_FUNC stop( VMARG );
};


}
}

#endif // GTK_SPINNER_HPP
