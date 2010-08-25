#ifndef GTK_SCROLLBAR_HPP
#define GTK_SCROLLBAR_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Scrollbar
 */
class Scrollbar
    :
    public Gtk::CoreGObject
{
public:

    static void modInit( Falcon::Module* );

#if GTK_CHECK_VERSION( 3, 0, 0 )
    static FALCON_FUNC init( VMARG );
#endif

};


} // Gtk
} // Falcon

#endif // !GTK_SCROLLBAR_HPP
