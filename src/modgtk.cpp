/**
 *  \file modgtk.cpp
 */

#include "modgtk.hpp"

#include "g_Object.hpp"
#include "g_ParamSpec.hpp"

#include "gdk_DragContext.hpp"
#include "gdk_EventButton.hpp"
#include "gdk_Pixbuf.hpp"

#include "gtk_enums.hpp"

#if GTK_MINOR_VERSION >= 6
#include "gtk_AboutDialog.hpp"
#endif
#include "gtk_Action.hpp"
#include "gtk_Adjustment.hpp"
#include "gtk_Alignment.hpp"
#include "gtk_Arrow.hpp"
#include "gtk_AspectFrame.hpp"
#include "gtk_Bin.hpp"
#include "gtk_Box.hpp"
#include "gtk_Button.hpp"
#include "gtk_ButtonBox.hpp"
#include "gtk_CheckButton.hpp"
#include "gtk_ComboBox.hpp"
#include "gtk_ComboBoxEntry.hpp"
#include "gtk_Container.hpp"
#include "gtk_Dialog.hpp"
#include "gtk_Entry.hpp"
#include "gtk_EntryBuffer.hpp"
#include "gtk_EventBox.hpp"
#include "gtk_Expander.hpp"
#include "gtk_Fixed.hpp"
#include "gtk_Frame.hpp"
#include "gtk_HBox.hpp"
#include "gtk_HButtonBox.hpp"
#include "gtk_HPaned.hpp"
#include "gtk_Image.hpp"
#include "gtk_Label.hpp"
#include "gtk_Layout.hpp"
#include "gtk_Main.hpp"
#include "gtk_Misc.hpp"
#include "gtk_Object.hpp"
#include "gtk_Paned.hpp"
#include "gtk_RadioAction.hpp"
#include "gtk_RadioButton.hpp"
#include "gtk_Requisition.hpp"
#include "gtk_SpinButton.hpp"
#include "gtk_Stock.hpp"
#include "gtk_Table.hpp"
#include "gtk_TextBuffer.hpp"
#include "gtk_TextIter.hpp"
#include "gtk_TextMark.hpp"
#include "gtk_TextTag.hpp"
#include "gtk_TextTagTable.hpp"
#include "gtk_TextView.hpp"
#include "gtk_ToggleAction.hpp"
#include "gtk_ToggleButton.hpp"
#include "gtk_VBox.hpp"
#include "gtk_VButtonBox.hpp"
#include "gtk_VPaned.hpp"
#include "gtk_Widget.hpp"
#include "gtk_Window.hpp"


FALCON_MODULE_DECL
{
    #define FALCON_DECLARE_MODULE self

    Falcon::Module* self = new Falcon::Module();
    self->name( "gtk" );
    self->language( "en_US" );
    self->engineVersion( FALCON_VERSION_NUM );
    self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

    #include "modgtk_st.hpp"

    /*
     *  load glib
     */

    Falcon::Glib::Object::modInit( self );
    Falcon::Glib::ParamSpec::modInit( self );

    /*
     *  load gdk
     */

    Falcon::Gdk::DragContext::modInit( self );
    Falcon::Gdk::EventButton::modInit( self );
    Falcon::Gdk::Pixbuf::modInit( self );

    /*
     *  load enums
     */
    Falcon::Gtk::Enums::modInit( self );

    /*
     *  setup the classes
     */

    // not GObject based //

    Falcon::Gtk::Main::modInit( self );
    Falcon::Gtk::Requisition::modInit( self );
    Falcon::Gtk::Signal::modInit( self );
    Falcon::Gtk::Stock::modInit( self );
    Falcon::Gtk::TextIter::modInit( self );

    // GObject based //

    Falcon::Gtk::Action::modInit( self );
        Falcon::Gtk::ToggleAction::modInit( self );
            Falcon::Gtk::RadioAction::modInit( self );
#if GTK_VERSION_MINOR >= 18
    Falcon::Gtk::EntryBuffer::modInit( self );
#endif
    Falcon::Gtk::Object::modInit( self );
        Falcon::Gtk::Adjustment::modInit( self );
        Falcon::Gtk::Widget::modInit( self );
            Falcon::Gtk::Container::modInit( self );
                Falcon::Gtk::Bin::modInit( self );
                    Falcon::Gtk::Alignment::modInit( self );
                    Falcon::Gtk::Button::modInit( self );
                        Falcon::Gtk::ToggleButton::modInit( self );
                            Falcon::Gtk::CheckButton::modInit( self );
                                Falcon::Gtk::RadioButton::modInit( self );
                    Falcon::Gtk::ComboBox::modInit( self );
                        Falcon::Gtk::ComboBoxEntry::modInit( self );
                    Falcon::Gtk::EventBox::modInit( self );
                    Falcon::Gtk::Expander::modInit( self );
                    Falcon::Gtk::Frame::modInit( self );
                        Falcon::Gtk::AspectFrame::modInit( self );
                    Falcon::Gtk::Window::modInit( self );
                        Falcon::Gtk::Dialog::modInit( self );
#if GTK_MINOR_VERSION >= 6
                            Falcon::Gtk::AboutDialog::modInit( self );
#endif
                Falcon::Gtk::Box::modInit( self );
                    Falcon::Gtk::ButtonBox::modInit( self );
                        Falcon::Gtk::HButtonBox::modInit( self );
                        Falcon::Gtk::VButtonBox::modInit( self );
                    Falcon::Gtk::HBox::modInit( self );
                    Falcon::Gtk::VBox::modInit( self );
                Falcon::Gtk::Fixed::modInit( self );
                Falcon::Gtk::Layout::modInit( self );
                Falcon::Gtk::Paned::modInit( self );
                    Falcon::Gtk::HPaned::modInit( self );
                    Falcon::Gtk::VPaned::modInit( self );
                Falcon::Gtk::Table::modInit( self );
                Falcon::Gtk::TextView::modInit( self );
            Falcon::Gtk::Entry::modInit( self );
                Falcon::Gtk::SpinButton::modInit( self );
            Falcon::Gtk::Misc::modInit( self );
                Falcon::Gtk::Arrow::modInit( self );
                Falcon::Gtk::Image::modInit( self );
                Falcon::Gtk::Label::modInit( self );
    Falcon::Gtk::TextBuffer::modInit( self );
    Falcon::Gtk::TextMark::modInit( self );
    Falcon::Gtk::TextTag::modInit( self );
    Falcon::Gtk::TextTagTable::modInit( self );

    return self;
}


namespace Falcon {
namespace Gtk {


FALCON_FUNC abstract_init( VMARG )
{
    MYSELF;
    // check that a derived class has stuffed an object here
    if ( !self->getGObject() )
    {
        throw_gtk_error( e_abstract_class, FAL_STR( gtk_e_abstract_class_ ) );
    }
}


CoreGObject::CoreGObject( const Falcon::CoreClass* cls, const GObject* gobj )
    :
    Falcon::FalconObject( cls ),
    m_obj( (GObject*) gobj )
{
    incref();
}


CoreGObject::CoreGObject( const CoreGObject& other )
    :
    Falcon::FalconObject( other ),
    m_obj( other.m_obj )
{
    incref();
}


CoreGObject* CoreGObject::clone() const
{
    return new CoreGObject( *this );
}


void CoreGObject::setGObject( const GObject* obj )
{
    assert( !m_obj && obj );
    m_obj = (GObject*) obj;
    incref();
}


bool CoreGObject::getProperty( const Falcon::String& s, Falcon::Item& it ) const
{
    AutoCString cstr( s );
    GarbageLock* lock = (GarbageLock*) g_object_get_data( m_obj, cstr.c_str() );
    if ( lock )
        it = lock->item();
    else
        return defaultProperty( s, it );
    return true;
}


bool CoreGObject::setProperty( const Falcon::String& s, const Falcon::Item& it )
{
    AutoCString cstr( s );
    g_object_set_data_full( m_obj, cstr.c_str(),
            new GarbageLock( it ), &CoreGObject::release_lock );
    return true;
}


GObject* CoreGObject::add_slots( GObject* obj )
{
    if ( !g_object_get_data( obj, "__signals" ) )
    {
        g_object_set_data_full( obj, "__signals",
            (gpointer) new Falcon::CoreSlot( "" ), &CoreGObject::release_slots );
    }
    return obj;
}


void CoreGObject::release_slots( gpointer cs )
{
    ((Falcon::CoreSlot*)cs)->clear();
    ((Falcon::CoreSlot*)cs)->decref();
}


void CoreGObject::get_signal( const char* signame, const void* cb, Falcon::VMachine* vm )
{
    CoreGObject* self = Falcon::dyncast<CoreGObject*>( vm->self().asObjectSafe() );

    vm->retval( new Gtk::Signal( vm->findWKI( "GtkSignal" )->asClass(),
            self->m_obj, signame, cb ) );
}


void CoreGObject::trigger_slot( GObject* obj, const char* signame,
                            const char* cbname, Falcon::VMachine* vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( signame, false );

    if ( !cs || cs->empty() )
        return;

    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();
        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( cbname, it ) )
            {
                printf( "[%s] invalid callback (expected callable)\n", cbname );
                return;
            }
        }
        vm->callItem( it, 0 );
        iter.next();
    }
    while ( iter.hasCurrent() );
}


GPtrArray* CoreGObject::get_locks( GObject* obj )
{
    GPtrArray* arr;
    if ( !( arr = (GPtrArray*) g_object_get_data( obj, "__locks" ) ) )
    {
        arr = g_ptr_array_new_with_free_func( &CoreGObject::release_lock );

        g_object_set_data_full( obj, "__locks", (gpointer) arr,
            &CoreGObject::release_locks );
    }
    return arr;
}


void CoreGObject::release_lock( gpointer glock )
{
    delete (GarbageLock*) glock;
}


void CoreGObject::release_locks( gpointer arr )
{
    g_ptr_array_free( (GPtrArray*) arr, TRUE );
}


GarbageLock* CoreGObject::lockItem( GObject* obj, const Falcon::Item& it )
{
    GarbageLock* lock = new Falcon::GarbageLock( it );
    GPtrArray* arr = get_locks( obj );
    g_ptr_array_add( arr, (gpointer) lock );
    return lock;
}


Signal::Signal( const Falcon::CoreClass* cls,
                const GObject* gobj, const char* name, const void* cb )
    :
    Falcon::FalconObject( cls ),
    m_obj( (GObject*) gobj ),
    m_name( (char*) name ),
    m_cb( (void*) cb )
{
    incref();
}


bool Signal::getProperty( const Falcon::String& s, Falcon::Item& it ) const
{
    if ( s == "name" )
        it = Falcon::String( m_name );
    else
        return defaultProperty( s, it );
    return true;
}


bool Signal::setProperty( const Falcon::String&, const Falcon::Item& )
{
    return false;
}


Falcon::CoreObject* Signal::factory( const Falcon::CoreClass* cls, void* gobj, bool )
{
    return new Signal( cls, (GObject*) gobj, NULL, NULL );
}


void Signal::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Signal = mod->addClass( "GtkSignal" );

    c_Signal->setWKS( true );
    //c_Signal->getClassDef()->factory( &Signal::factory );

    mod->addClassMethod( c_Signal, "connect", &Signal::connect );
}


FALCON_FUNC Signal::connect( VMARG )
{
    Item* cb = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !cb || cb->isNil() || !( cb->isCallable() || cb->isComposed() ) )
        throw_inv_params( "C" );
#endif
    Signal* self = Falcon::dyncast<Signal*>( vm->self().asObjectSafe() );
    GET_OBJ( self );
    GET_SIGNALS( _obj );

    Falcon::CoreSlot* cs = _signals->getChild( self->m_name, true );
    cs->append( *cb );

    CoreGObject::lockItem( _obj, *cb );

    g_signal_connect( G_OBJECT( _obj ), self->m_name,
                      G_CALLBACK( self->m_cb ), vm );
}


uint32
getGCharArray( const Falcon::CoreArray* arr,
        gchar** strings,
        Falcon::AutoCString** temp )
{
    const uint32 num = arr->length();

    if ( !num ) return 0;

    uint32 i = 0;
    *strings = (gchar*) memAlloc( sizeof( gchar* ) * ( num + 1 ) );
    *temp = (AutoCString*) memAlloc( sizeof( AutoCString ) * num );
    strings[ num ] = NULL;
    Item s;

    for ( ; i < num; ++i )
    {
        s = arr->at( i );
#ifndef NO_PARAMETER_CHECK
        if ( !s.isString() )
            throw_inv_params( "S" );
#endif
        (*temp)[i].set( s.asString() );
        strings[i] = (gchar*) (*temp)[i].c_str();
    }

    return num;
}


/*#
    @class GtkError
    @brief Error generated by falcon-gtk
    @optparam code numeric error code
    @optparam description textual description of the error code
    @optparam extra descriptive message explaining the error conditions
    @from Error code, description, extra

    See the Error class in the core module.
*/
FALCON_FUNC GtkError_init( VMARG )
{
    MYSELF;

    if ( !self->getUserData() )
        self->setUserData( new Falcon::Gtk::GtkError );

    Falcon::core::Error_init( vm );
}


} // Gtk
} // Falcon
