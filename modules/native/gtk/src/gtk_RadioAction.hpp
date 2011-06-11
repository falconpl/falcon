#ifndef GTK_RADIOACTION_HPP
#define GTK_RADIOACTION_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::RadioAction
 */
class RadioAction
    :
    public Gtk::CoreGObject
{
public:

    RadioAction( const Falcon::CoreClass*, const GtkRadioAction* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_changed( VMARG );

    static void on_changed( GtkRadioAction*, GtkRadioAction*, gpointer );

//     static FALCON_FUNC get_group( VMARG );

//     static FALCON_FUNC set_group( VMARG );

    static FALCON_FUNC get_current_value( VMARG );

    static FALCON_FUNC set_current_value( VMARG );


};


} // Gtk
} // Falcon

#endif // !GTK_RADIOACTION_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
