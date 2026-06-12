import os
import json
from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP, func
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_PATH = "./db/webfingerprint.db"
DB_URL = f"sqlite:///{DATABASE_PATH}"

Base = declarative_base()


class Fingerprint(Base):
    __tablename__ = 'fingerprints'

    id = Column(Integer, primary_key=True, autoincrement=True)
    website = Column(String, nullable=False)
    website_index = Column(Integer, nullable=False)
    trace_data = Column(Text, nullable=False)
    timestamp = Column(TIMESTAMP, default=func.current_timestamp())

    def __repr__(self):
        return f"<Fingerprint(id={self.id}, website='{self.website}')>"


class CollectionStats(Base):
    __tablename__ = 'collection_stats'

    website = Column(String, primary_key=True)
    traces_collected = Column(Integer, default=0)

    def __repr__(self):
        return f"<CollectionStats(website='{self.website}', traces_collected={self.traces_collected})>"


class Database:
    def __init__(self, websites):
        self.engine = create_engine(DB_URL)
        self.Session = sessionmaker(bind=self.engine)
        self.websites = websites

    def init_database(self):
        """Initialize the database with required tables"""
        Base.metadata.create_all(self.engine)

        session = self.Session()
        try:
            for website in self.websites:
                existing = session.query(CollectionStats).filter_by(
                    website=website).first()
                if not existing:
                    session.add(CollectionStats(
                        website=website, traces_collected=0))
            session.commit()
            print(f"Database initialized at: {os.path.abspath(DATABASE_PATH)}")
        except Exception as e:
            session.rollback()
            print(f"Error initializing database: {str(e)}")
        finally:
            session.close()

    def save_trace(self, website, site_idx, trace_data):
        """Save a single trace to the database"""
        # Debug: print the trace data
        print(f"\nDEBUG - DATABASE - Trace data type: {type(trace_data)}")
        print(
            f"DEBUG - DATABASE - Trace data sample: {str(trace_data)[:200]}...")

        trace_json = json.dumps(trace_data)
        session = self.Session()

        try:
            session.add(Fingerprint(
                website=website,
                website_index=site_idx,
                trace_data=trace_json
            ))

            stats = session.query(CollectionStats).filter_by(
                website=website).first()
            if stats:
                stats.traces_collected += 1

            session.commit()
            print(f"Saved trace for {website}")
            return True
        except Exception as e:
            session.rollback()
            print(f"Error saving trace to database: {str(e)}")
            return False
        finally:
            session.close()

    def get_traces_collected(self):
        """Get the count of traces collected for each website"""
        session = self.Session()
        try:
            stats = session.query(CollectionStats).all()
            results = {stat.website: stat.traces_collected for stat in stats}
            return results
        except Exception as e:
            print(f"Error retrieving trace counts: {str(e)}")
            return {website: 0 for website in self.websites}
        finally:
            session.close()

    def export_to_json(self, output_path="dataset.json"):
        """Export the database contents to a JSON file"""
        session = self.Session()
        try:
            print(f"\nDEBUG - export_to_json: Retrieving fingerprints from database")
            fingerprints = session.query(Fingerprint).order_by(
                Fingerprint.website, Fingerprint.id).all()

            print(
                f"DEBUG - export_to_json: Found {len(fingerprints)} fingerprints")

            # List all found fingerprints
            for idx, fp in enumerate(fingerprints):
                print(
                    f"DEBUG - fingerprint {idx+1}: website={fp.website}, index={fp.website_index}, trace_len={len(fp.trace_data) if fp.trace_data else 0}")

            dataset = []
            for fp in fingerprints:
                try:
                    print(
                        f"DEBUG - Processing fingerprint for {fp.website} index {fp.website_index}")
                    print(
                        f"DEBUG - Trace data (first 50 chars): {fp.trace_data[:50]}...")
                    trace_data = json.loads(fp.trace_data)
                    dataset.append({
                        'website': fp.website,
                        'website_index': fp.website_index,
                        'trace_data': trace_data
                    })
                    print(
                        f"DEBUG - Successfully processed fingerprint for {fp.website}")
                except Exception as e:
                    print(
                        f"Error parsing trace data for {fp.website}: {str(e)}")

            with open(output_path, 'w') as f:
                json.dump(dataset, f)

            print(f"Exported {len(dataset)} records to {output_path}")
        except Exception as e:
            print(f"Error exporting to JSON: {str(e)}")
        finally:
            session.close()

    def export_database(self):
        """Export the entire database content for backup or migration"""
        session = self.Session()
        try:
            fingerprints = session.query(Fingerprint).order_by(
                Fingerprint.website, Fingerprint.id).all()

            # Export logic here
            return True
        except Exception as e:
            print(f"Error exporting database: {str(e)}")
            return False
        finally:
            session.close()


