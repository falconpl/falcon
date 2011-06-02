#ifndef GTK_OPTIONMENU_HPP
#define GTK_OPTIONMENU_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::OptionMenu
 */
class OptionMenu
    :
    public Gtk::CoreGObject
{
public:

    OptionMenu( const Falcon::CoreClass*, const GtkOptionMenu* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_changed( VMARG );

    static void on_changed( GtkOptionMenu*, gpointer );

    static FALCON_FUNC get_menu( VMARG );

    static FALCON_FUNC set_menu( VMARG );

    static FALCON_FUNC remove_menu( VMARG );

    static FALCON_FUNC set_history( VMARG );

    static FALCON_FUNC get_history( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_OPTIONMENU_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
